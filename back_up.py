from dataclasses import dataclass, field
import math
import numpy as np
import time
from typing import List, Tuple, Dict, Any

# ----------------------------- Basic datatypes (bit-accurate) -----------------------------

@dataclass
class TensorShape:
    dims: Tuple[int, ...]

@dataclass
class Tensor:
    name: str
    shape: TensorShape
    size_bits: int
    device: str = "dram"  # 'dram' or 'rram'

    @property
    def size_bytes(self) -> int:
        return (self.size_bits + 7) // 8

# ----------------------------- Memory models (bit units) -----------------------------

@dataclass
class MemoryDevice:
    name: str
    capacity_bits: int
    read_bw_bits_per_cycle: int  # bits per cycle read bandwidth
    write_bw_bits_per_cycle: int
    read_energy_per_bit: float   # nJ per bit read
    write_energy_per_bit: float  # nJ per bit write
    access_latency_cycles: int
    used_bits: int = 0

    def can_allocate(self, size_bits: int) -> bool:
        return self.used_bits + size_bits <= self.capacity_bits

    def allocate(self, size_bits: int) -> bool:
        if self.can_allocate(size_bits):
            self.used_bits += size_bits
            return True
        return False

    def free(self, size_bits: int) -> None:
        self.used_bits = max(0, self.used_bits - size_bits)

# ----------------------------- Compute unit model -----------------------------

@dataclass
class ComputeUnit:
    macs_per_cycle: int
    energy_per_mac_nj: float  # energy per MAC in nJ

# ------------ ACU model (for complex elementwise & nonlinear ops) -------------

@dataclass
class ACU:
    # throughput in elements per cycle (how many scalar elements ACU can process per cycle)
    throughput_elements_per_cycle: int
    # base energy per processed element (nJ)
    energy_per_element_nj: float
    # latency overhead (cycles) per call
    call_latency_cycles: int = 1

    # per-op cost multiplier (some ops cost more than a simple element op)
    op_cost_multiplier: Dict[str, float] = field(default_factory=lambda: {
        'NEG': 0.2,
        'DIV': 2.0,
        'EXP': 5.0,
        'GELU': 4.0,
        'SILU': 3.0,
        'SOFTMAX': 6.0,
        'LAYERNORM': 6.0,
    })

    def op_cycles_and_energy(self, op_name: str, elements: int) -> Tuple[int, float]:
        m = self.op_cost_multiplier.get(op_name.upper(), 1.0)
        cycles = math.ceil(elements / self.throughput_elements_per_cycle) + self.call_latency_cycles
        energy = elements * self.energy_per_element_nj * m
        return cycles, energy

# ----------------------------- Operation models -----------------------------

class Op:
    def __init__(self, name: str):
        self.name = name

    def required_tensors(self) -> List[str]:
        return []

class MatMul(Op):
    def __init__(self, A: str, B: str, C: str):
        super().__init__(f"MatMul:{A}x{B}->{C}")
        self.A = A
        self.B = B
        self.C = C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        a = shapes[self.A].dims
        b = shapes[self.B].dims
        M, K = a
        K2, N = b
        assert K == K2
        return M * N * K

    def required_tensors(self) -> List[str]:
        return [self.A, self.B]

class PatchEmbed(Op):
    def __init__(self, input_img: str, patch_w: int, patch_h: int, out_name: str, weight_name: str):
        super().__init__(f"PatchEmbed:{input_img}->{out_name}")
        self.input_img = input_img
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.out_name = out_name
        self.weight_name = weight_name

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        img_shape = shapes[self.input_img].dims  # (C,H,W)
        C, H, W = img_shape
        n_patches_h = H // self.patch_h
        n_patches_w = W // self.patch_w
        num_patches = n_patches_h * n_patches_w
        patch_size = C * self.patch_h * self.patch_w
        out_dim = shapes[self.weight_name].dims[1]
        return num_patches * patch_size * out_dim

    def required_tensors(self) -> List[str]:
        return [self.input_img, self.weight_name]

# Elementwise unary ops (NEG, EXP, GELU, SILU) and binary (DIV)
class UnaryOp(Op):
    def __init__(self, kind: str, A: str, C: str):
        super().__init__(f"{kind}:{A}->{C}")
        self.kind = kind.upper()
        self.A = A
        self.C = C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        ashape = shapes[self.A].dims
        return int(np.prod(ashape))

    def required_tensors(self) -> List[str]:
        return [self.A]

class BinaryOp(Op):
    def __init__(self, kind: str, A: str, B: str, C: str):
        super().__init__(f"{kind}:{A},{B}->{C}")
        self.kind = kind.upper()
        self.A = A
        self.B = B
        self.C = C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        ash = shapes[self.A].dims
        bsh = shapes[self.B].dims
        assert ash == bsh
        return int(np.prod(ash))

    def required_tensors(self) -> List[str]:
        return [self.A, self.B]

class SoftmaxOp(Op):
    def __init__(self, input_tensor: str, axis: int, output_tensor: str):
        super().__init__(f"Softmax:{input_tensor}-> {output_tensor} axis={axis}")
        self.input = input_tensor
        self.axis = axis
        self.output = output_tensor

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        dims = shapes[self.input].dims
        total = int(np.prod(dims))
        return total

    def required_tensors(self) -> List[str]:
        return [self.input]

class LayerNorm(Op):
    def __init__(self, A: str, C: str):
        super().__init__(f"LayerNorm:{A}->{C}")
        self.A = A
        self.C = C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        # layernorm per element involves a few ops (mean, var, normalize)
        ashape = shapes[self.A].dims
        return int(np.prod(ashape))

    def required_tensors(self) -> List[str]:
        return [self.A]

class Attention(Op):
    def __init__(self, Q: str, K: str, V: str, out: str):
        super().__init__(f"Attention:{Q},{K},{V}->{out}")
        self.Q = Q
        self.K = K
        self.V = V
        self.out = out

    def required_tensors(self) -> List[str]:
        return [self.Q, self.K, self.V]

# ----------------------------- Simple model description -----------------------------

class Model:
    def __init__(self):
        self.tensors: Dict[str, Tensor] = {}
        self.shapes: Dict[str, TensorShape] = {}
        self.ops: List[Op] = []

    def add_tensor(self, name: str, shape: Tuple[int, ...], bits_per_element=32, device='dram'):
        shape_obj = TensorShape(shape)
        size_bits = int(np.prod(shape) * bits_per_element)
        self.tensors[name] = Tensor(name, shape_obj, size_bits, device)
        self.shapes[name] = shape_obj

    def add_op(self, op: Op):
        self.ops.append(op)

# ----------------------------- Compiler (mapper + scheduler) -----------------------------

class SimpleCompiler:
    def __init__(self, model: Model, rram: MemoryDevice, dram: MemoryDevice, cu: ComputeUnit,
                 bits_per_element=32, tile_K=256, tile_M=64, tile_N=64):
        self.model = model
        self.rram = rram
        self.dram = dram
        self.cu = cu
        self.tile_K = tile_K
        self.tile_M = tile_M
        self.tile_N = tile_N
        self.bpe_bits = bits_per_element

    def place(self):
        placements = {}
        for tname, tensor in self.model.tensors.items():
            if tname.startswith('W') or tensor.device == 'rram':
                if self.rram.allocate(tensor.size_bits):
                    placements[tname] = 'rram'
                    tensor.device = 'rram'
                else:
                    placements[tname] = 'dram'
                    tensor.device = 'dram'
            else:
                if self.dram.allocate(tensor.size_bits):
                    placements[tname] = 'dram'
                    tensor.device = 'dram'
                else:
                    placements[tname] = 'rram'
                    tensor.device = 'rram'
        return placements

    def compile(self) -> List[Dict[str, Any]]:
        placements = self.place()
        schedule = []
        for op in self.model.ops:
            # decompose high-level ops into micro-steps that the simulator/architecture understand
            if isinstance(op, MatMul):
                dims = self.model.shapes[op.A].dims
                M, K = dims[-2], dims[-1]
                K2, N = self.model.shapes[op.B].dims
                assert K == K2
                for m0 in range(0, M, self.tile_M):
                    msize = min(self.tile_M, M - m0)
                    for n0 in range(0, N, self.tile_N):
                        nsize = min(self.tile_N, N - n0)
                        for k0 in range(0, K, self.tile_K):
                            ksize = min(self.tile_K, K - k0)
                            item = {
                                'op': op,
                                'type': 'matmul_tile',
                                'm0': m0, 'n0': n0, 'k0': k0,
                                'msize': msize, 'nsize': nsize, 'ksize': ksize,
                                'A_dev': self.model.tensors[op.A].device,
                                'B_dev': self.model.tensors[op.B].device
                            }
                            schedule.append(item)
            elif isinstance(op, PatchEmbed):
                schedule.append({'op': op, 'type': 'patch_embed'})
            elif isinstance(op, LayerNorm):
                schedule.append({'op': op, 'type': 'layernorm'})
            elif isinstance(op, Attention):
                # decompose attention into: QK^T (matmul), softmax, softmax*V (matmul)
                # shapes: Q (L,d), K (L,d), V (L,d) -> QK^T: (L,L)
                schedule.append({'op': op, 'type': 'attention'})
            elif isinstance(op, SoftmaxOp):
                schedule.append({'op': op, 'type': 'softmax'})
            elif isinstance(op, UnaryOp):
                schedule.append({'op': op, 'type': 'unary'})
            elif isinstance(op, BinaryOp):
                schedule.append({'op': op, 'type': 'binary'})
            else:
                schedule.append({'op': op, 'type': 'unknown'})
        return schedule

# ----------------------------- Simulator core (bit units + ACU) -----------------------------

@dataclass
class Stats:
    cycles: int = 0
    energy_nj: float = 0.0
    macs: int = 0
    bits_read: int = 0
    bits_written: int = 0
    breakdown: Dict[str, float] = field(default_factory=dict)

class Simulator:
    def __init__(self, model: Model, schedule: List[Dict[str, Any]], rram: MemoryDevice, dram: MemoryDevice, cu: ComputeUnit, acu: ACU, bits_per_element=32):
        self.model = model
        self.schedule = schedule
        self.rram = rram
        self.dram = dram
        self.cu = cu
        self.acu = acu
        self.stats = Stats()
        self.bpe_bits = bits_per_element

    def _mem_read_cost(self, dev: MemoryDevice, size_bits: int):
        bw_cycles = math.ceil(size_bits / dev.read_bw_bits_per_cycle) if dev.read_bw_bits_per_cycle>0 else 0
        cycles = dev.access_latency_cycles + bw_cycles
        energy = size_bits * dev.read_energy_per_bit
        return cycles, energy

    def _mem_write_cost(self, dev: MemoryDevice, size_bits: int):
        bw_cycles = math.ceil(size_bits / dev.write_bw_bits_per_cycle) if dev.write_bw_bits_per_cycle>0 else 0
        cycles = dev.access_latency_cycles + bw_cycles
        energy = size_bits * dev.write_energy_per_bit
        return cycles, energy

    def _compute_cost(self, macs: int):
        cycles = math.ceil(macs / self.cu.macs_per_cycle)
        energy = macs * self.cu.energy_per_mac_nj
        return cycles, energy

    def run(self):
        for item in self.schedule:
            ttype = item['type']
            op = item.get('op')
            if ttype == 'matmul_tile':
                m = item['msize']; n = item['nsize']; k = item['ksize']
                A_tile_bits = m * k * self.bpe_bits
                B_tile_bits = k * n * self.bpe_bits
                devA = self.model.tensors[op.A].device
                devB = self.model.tensors[op.B].device
                memA = self.rram if devA=='rram' else self.dram
                memB = self.rram if devB=='rram' else self.dram
                cA, eA = self._mem_read_cost(memA, A_tile_bits)
                cB, eB = self._mem_read_cost(memB, B_tile_bits)
                macs = m * n * k
                cc, ec = self._compute_cost(macs)
                C_tile_bits = m * n * self.bpe_bits
                cW, eW = self._mem_write_cost(self.dram, C_tile_bits)
                cycles = max(cA + cB, cc) + cW
                energy = eA + eB + ec + eW
                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.macs += macs
                self.stats.bits_read += (A_tile_bits + B_tile_bits)
                self.stats.bits_written += C_tile_bits
                self.stats.breakdown['matmul'] = self.stats.breakdown.get('matmul',0)+energy
            elif ttype == 'patch_embed':
                op: PatchEmbed = op
                img_shape = self.model.shapes[op.input_img].dims
                Cc, H, W = img_shape
                nph = H // op.patch_h
                npw = W // op.patch_w
                num_patches = nph * npw
                patch_size = Cc * op.patch_h * op.patch_w
                out_dim = self.model.shapes[op.weight_name].dims[1]
                weight_bits = patch_size * out_dim * self.bpe_bits
                cW, eW = self._mem_read_cost(self.rram, weight_bits)
                img_bits = self.model.tensors[op.input_img].size_bits
                cI, eI = self._mem_read_cost(self.dram, img_bits)
                macs = num_patches * patch_size * out_dim
                cc, ec = self._compute_cost(macs)
                out_bits = num_patches * out_dim * self.bpe_bits
                cOut, eOut = self._mem_write_cost(self.dram, out_bits)
                cycles = max(cW + cI, cc) + cOut
                energy = eW + eI + ec + eOut
                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.macs += macs
                self.stats.bits_read += (weight_bits + img_bits)
                self.stats.bits_written += out_bits
                self.stats.breakdown['patch_embed'] = self.stats.breakdown.get('patch_embed',0)+energy
            elif ttype == 'layernorm':
                op: LayerNorm = op
                elems = int(np.prod(self.model.shapes[op.A].dims))
                cycles, energy = self.acu.op_cycles_and_energy('LAYERNORM', elems)
                input_bits = self.model.tensors[op.A].size_bits
                read_c, read_e = self._mem_read_cost(self.dram, input_bits) if self.model.tensors[op.A].device=='dram' else self._mem_read_cost(self.rram, input_bits)
                out_bits = self.model.tensors[op.C].size_bits if op.C in self.model.tensors else input_bits
                write_c, write_e = self._mem_write_cost(self.dram, out_bits)
                total_cycles = max(read_c, cycles) + write_c
                total_energy = read_e + write_e + energy
                self.stats.cycles += total_cycles
                self.stats.energy_nj += total_energy
                self.stats.bits_read += input_bits
                self.stats.bits_written += out_bits
                self.stats.breakdown['layernorm'] = self.stats.breakdown.get('layernorm',0)+energy
            elif ttype == 'attention':
                op: Attention = op
                # get shapes
                Q_shape = self.model.shapes[op.Q].dims  # (L,d)
                L, d = Q_shape[-2], Q_shape[-1]
                # QK^T: (L,L) --> macs = L*L*d
                macs_qk = L * L * d
                c_qk, e_qk = self._compute_cost(macs_qk)
                # mem cost: read Q and K
                q_bits = self.model.tensors[op.Q].size_bits
                k_bits = self.model.tensors[op.K].size_bits
                rq_c, rq_e = self._mem_read_cost(self.dram, q_bits) if self.model.tensors[op.Q].device=='dram' else self._mem_read_cost(self.rram, q_bits)
                rk_c, rk_e = self._mem_read_cost(self.dram, k_bits) if self.model.tensors[op.K].device=='dram' else self._mem_read_cost(self.rram, k_bits)
                # produce attention scores matrix (LxL) stored in dram
                scores_bits = L * L * self.bpe_bits
                wsc_c, wsc_e = self._mem_write_cost(self.dram, scores_bits)
                # softmax on scores (ACU)
                soft_cycles, soft_energy = self.acu.op_cycles_and_energy('SOFTMAX', L * L)
                # read V and compute softmax * V: macs = L*L*d  (dense)
                v_bits = self.model.tensors[op.V].size_bits
                rv_c, rv_e = self._mem_read_cost(self.dram, v_bits) if self.model.tensors[op.V].device=='dram' else self._mem_read_cost(self.rram, v_bits)
                macs_sv = L * L * d
                c_sv, e_sv = self._compute_cost(macs_sv)
                out_bits = L * d * self.bpe_bits
                write_out_c, write_out_e = self._mem_write_cost(self.dram, out_bits)
                # schedule cycles: read Q/K + compute qk + write scores + softmax + read V + compute sv + write out
                cycles = max(rq_c + rk_c + c_qk, soft_cycles) + wsc_c + rv_c + c_sv + write_out_c
                energy = rq_e + rk_e + e_qk + wsc_e + soft_energy + rv_e + e_sv + write_out_e
                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.macs += (macs_qk + macs_sv)
                self.stats.bits_read += (q_bits + k_bits + v_bits)
                self.stats.bits_written += (scores_bits + out_bits)
                self.stats.breakdown['attention'] = self.stats.breakdown.get('attention',0)+energy
            elif ttype == 'softmax':
                op: SoftmaxOp = op
                elems = int(np.prod(self.model.shapes[op.input].dims))
                cycles, energy = self.acu.op_cycles_and_energy('SOFTMAX', elems)
                input_bits = self.model.tensors[op.input].size_bits
                read_c, read_e = self._mem_read_cost(self.dram, input_bits)
                out_bits = self.model.tensors[op.output].size_bits if op.output in self.model.tensors else input_bits
                write_c, write_e = self._mem_write_cost(self.dram, out_bits)
                total_cycles = max(read_c, cycles) + write_c
                total_energy = read_e + write_e + energy
                self.stats.cycles += total_cycles
                self.stats.energy_nj += total_energy
                self.stats.bits_read += input_bits
                self.stats.bits_written += out_bits
                self.stats.breakdown['softmax'] = self.stats.breakdown.get('softmax',0)+energy
            elif ttype == 'unary':
                op: UnaryOp = op
                elems = int(np.prod(self.model.shapes[op.A].dims))
                cycles, energy = self.acu.op_cycles_and_energy(op.kind, elems)
                input_bits = self.model.tensors[op.A].size_bits
                out_bits = self.model.tensors[op.C].size_bits if op.C in self.model.tensors else input_bits
                read_c, read_e = self._mem_read_cost(self.dram, input_bits) if self.model.tensors[op.A].device=='dram' else self._mem_read_cost(self.rram, input_bits)
                write_c, write_e = self._mem_write_cost(self.dram, out_bits)
                total_cycles = max(read_c, cycles) + write_c
                total_energy = read_e + write_e + energy
                self.stats.cycles += total_cycles
                self.stats.energy_nj += total_energy
                self.stats.bits_read += input_bits
                self.stats.bits_written += out_bits
                self.stats.breakdown[op.kind.lower()] = self.stats.breakdown.get(op.kind.lower(),0)+energy
            elif ttype == 'binary':
                op: BinaryOp = op
                elems = int(np.prod(self.model.shapes[op.A].dims))
                cycles, energy = self.acu.op_cycles_and_energy(op.kind, elems)
                a_bits = self.model.tensors[op.A].size_bits
                b_bits = self.model.tensors[op.B].size_bits
                read_c1, read_e1 = self._mem_read_cost(self.dram, a_bits) if self.model.tensors[op.A].device=='dram' else self._mem_read_cost(self.rram, a_bits)
                read_c2, read_e2 = self._mem_read_cost(self.dram, b_bits) if self.model.tensors[op.B].device=='dram' else self._mem_read_cost(self.rram, b_bits)
                out_bits = self.model.tensors[op.C].size_bits if op.C in self.model.tensors else a_bits
                write_c, write_e = self._mem_write_cost(self.dram, out_bits)
                total_cycles = max(read_c1 + read_c2, cycles) + write_c
                total_energy = read_e1 + read_e2 + write_e + energy
                self.stats.cycles += total_cycles
                self.stats.energy_nj += total_energy
                self.stats.bits_read += (a_bits + b_bits)
                self.stats.bits_written += out_bits
                self.stats.breakdown[op.kind.lower()] = self.stats.breakdown.get(op.kind.lower(),0)+energy
            else:
                self.stats.cycles += 1
        return self.stats

# ----------------------------- Transformer encoder builder & example -----------------------------

def build_transformer_encoder(num_layers=6, seq_len=196, embed_dim=768, bits_act=16, bits_weight=4):
    """Constructs a Model with `num_layers` transformer encoder layers.
    Default seq_len=196 corresponds to 14x14 patches for 224x224 with 16x16 patches.
    Layer structure per your spec: norm -> attention -> add -> norm -> map(FFN) -> add
    """
    model = Model()
    B = 1
    L = seq_len
    D = embed_dim

    # input tokens
    model.add_tensor('input_tokens', (B, L, D), bits_per_element=bits_act, device='dram')

    for l in range(num_layers):
        # layernorm 1
        model.add_tensor(f'norm1_out_{l}', (B, L, D), bits_per_element=bits_act, device='dram')
        model.add_op(LayerNorm(f'input_tokens' if l==0 else f'resid2_{l-1}', f'norm1_out_{l}'))

        # Q,K,V projections (weights stored in rram at low precision)
        model.add_tensor(f'W_q_{l}', (D, D), bits_per_element=bits_weight, device='rram')
        model.add_tensor(f'W_k_{l}', (D, D), bits_per_element=bits_weight, device='rram')
        model.add_tensor(f'W_v_{l}', (D, D), bits_per_element=bits_weight, device='rram')

        # projected Q,K,V (activations)
        model.add_tensor(f'Q_{l}', (B, L, D), bits_per_element=bits_act, device='dram')
        model.add_tensor(f'K_{l}', (B, L, D), bits_per_element=bits_act, device='dram')
        model.add_tensor(f'V_{l}', (B, L, D), bits_per_element=bits_act, device='dram')

        # implement projection as batched MatMul: (B*L, D) x (D, D)
        model.add_tensor(f'norm1_flat_{l}', (B*L, D), bits_per_element=bits_act, device='dram')
        model.add_op(MatMul(f'norm1_flat_{l}', f'W_q_{l}', f'Q_{l}'))
        model.add_op(MatMul(f'norm1_flat_{l}', f'W_k_{l}', f'K_{l}'))
        model.add_op(MatMul(f'norm1_flat_{l}', f'W_v_{l}', f'V_{l}'))

        # attention op (will be decomposed by compiler)
        model.add_tensor(f'attn_out_{l}', (B, L, D), bits_per_element=bits_act, device='dram')
        model.add_op(Attention(f'Q_{l}', f'K_{l}', f'V_{l}', f'attn_out_{l}'))

        # residual add
        model.add_tensor(f'resid1_{l}', (B, L, D), bits_per_element=bits_act, device='dram')
        model.add_op(BinaryOp('ADD', f'attn_out_{l}', f'norm1_out_{l}', f'resid1_{l}'))

        # layernorm 2
        model.add_tensor(f'norm2_out_{l}', (B, L, D), bits_per_element=bits_act, device='dram')
        model.add_op(LayerNorm(f'resid1_{l}', f'norm2_out_{l}'))

        # FFN (map): two linear layers with GELU
        model.add_tensor(f'W1_{l}', (D, D*4), bits_per_element=bits_weight, device='rram')
        model.add_tensor(f'W2_{l}', (D*4, D), bits_per_element=bits_weight, device='rram')
        model.add_tensor(f'ff1_{l}', (B*L, D*4), bits_per_element=bits_act, device='dram')
        model.add_tensor(f'ff1_act_{l}', (B*L, D*4), bits_per_element=bits_act, device='dram')
        model.add_tensor(f'ff2_{l}', (B*L, D), bits_per_element=bits_act, device='dram')

        model.add_op(MatMul(f'norm2_out_{l}', f'W1_{l}', f'ff1_{l}'))
        model.add_op(UnaryOp('GELU', f'ff1_{l}', f'ff1_act_{l}'))
        model.add_op(MatMul(f'ff1_act_{l}', f'W2_{l}', f'ff2_{l}'))

        # residual add 2
        model.add_tensor(f'resid2_{l}', (B, L, D), bits_per_element=bits_act, device='dram')
        model.add_op(BinaryOp('ADD', f'ff2_{l}', f'resid1_{l}', f'resid2_{l}'))

    # final output tensor name: resid2_{num_layers-1}
    return model

# ----------------------------- Example run -----------------------------

if __name__ == '__main__':
    # device parameters (converted to bits)
    rram = MemoryDevice(name='3D_RRAM', capacity_bits=32*1024*1024*8, read_bw_bits_per_cycle=1024*8, write_bw_bits_per_cycle=512*8,
                        read_energy_per_bit=0.0005/8, write_energy_per_bit=0.0008/8, access_latency_cycles=5)
    dram = MemoryDevice(name='3D_DRAM', capacity_bits=256*1024*1024*8, read_bw_bits_per_cycle=8192*8, write_bw_bits_per_cycle=4096*8,
                        read_energy_per_bit=0.002/8, write_energy_per_bit=0.003/8, access_latency_cycles=50)
    cu = ComputeUnit(macs_per_cycle=8192, energy_per_mac_nj=0.00015)
    acu = ACU(throughput_elements_per_cycle=256, energy_per_element_nj=0.0005, call_latency_cycles=2)

    # build a 6-layer encoder (seq_len default 196 for 14x14 patches)
    model = build_transformer_encoder(num_layers=6, seq_len=196, embed_dim=768, bits_act=16, bits_weight=4)

    compiler = SimpleCompiler(model, rram, dram, cu, bits_per_element=16, tile_K=256, tile_M=128, tile_N=128)
    schedule = compiler.compile()

    sim = Simulator(model, schedule, rram, dram, cu, acu, bits_per_element=16)

    t0 = time.time()
    stats = sim.run()
    t1 = time.time()

    print("Simulation result (6-layer Transformer encoder on hetero PIM + ACU):")
    print(f"Total cycles: {stats.cycles}")
    print(f"Total MACs: {stats.macs}")
    print(f"Total energy (nJ): {stats.energy_nj:.2f}")
    print(f"Bits read: {stats.bits_read}")
    print(f"Bits written: {stats.bits_written}")
    print(f"Bytes read (approx): {stats.bits_read/8:.1f}")
    print(f"Bytes written (approx): {stats.bits_written/8:.1f}")
    print(f"Runtime (s): {t1-t0:.3f}")

    freq_ghz = 1.0  # assume 1 GHz
    exec_time_s = stats.cycles / (freq_ghz * 1e9)
    energy_j = stats.energy_nj * 1e-9
    print(f"Estimated wall time @1GHz: {exec_time_s:.6f} s")
    print(f"Estimated energy (J): {energy_j:.6f} J")

    print('Breakdown:')
    for k,v in stats.breakdown.items():
        print(f'  {k}: {v:.2f} nJ')

    print('Notes:')
    print('- The compiler decomposes Attention into QK^T, Softmax, and Softmax*V, and the simulator maps these to PIM and ACU as appropriate.')
    print('- LayerNorm and elementwise/softmax ops run on ACU; matmuls run on PIM.')
    print('- You can tune per-device parameters to match target tech.')

# End of file
