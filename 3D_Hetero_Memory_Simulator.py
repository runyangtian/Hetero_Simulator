"""
3D Hetero Memory Simulator (Python)

Single-file prototype simulator for a heterogeneous 3D DRAM + 3D RRAM architecture.
Features:
- Simple model description (layers/ops)
- Memory device models: 3D RRAM (weights), 3D DRAM (activations)
- Compute unit model (MAC throughput, energy/op)
- Compiler: naive mapper & scheduler (tiling + placement)
- Cycle-accurate stepping and energy accounting (read/write, MACs)
- Example: patch embedding + simple attention-style matmul workload

This is meant as a starting engineering project. Extendable.

How to run:
- Save as 3D_Hetero_Memory_Simulator.py and run `python 3D_Hetero_Memory_Simulator.py`

"""

from dataclasses import dataclass, field
import math
import numpy as np
import time
from typing import List, Tuple, Dict, Any

# ----------------------------- Basic datatypes -----------------------------

@dataclass
class TensorShape:
    dims: Tuple[int, ...]

@dataclass
class Tensor:
    name: str
    shape: TensorShape
    size_bytes: int
    device: str = "dram"  # 'dram' or 'rram'

# ----------------------------- Memory models -----------------------------

@dataclass
class MemoryDevice:
    name: str
    capacity_bytes: int
    read_bw_bytes_per_cycle: int  # bytes per cycle read bandwidth
    write_bw_bytes_per_cycle: int
    read_energy_per_byte: float   # nJ per byte read
    write_energy_per_byte: float  # nJ per byte write
    access_latency_cycles: int
    used_bytes: int = 0

    def can_allocate(self, size_bytes: int) -> bool:
        return self.used_bytes + size_bytes <= self.capacity_bytes

    def allocate(self, size_bytes: int) -> bool:
        if self.can_allocate(size_bytes):
            self.used_bytes += size_bytes
            return True
        return False

    def free(self, size_bytes: int) -> None:
        self.used_bytes = max(0, self.used_bytes - size_bytes)

# ----------------------------- Compute unit model -----------------------------

@dataclass
class ComputeUnit:
    macs_per_cycle: int
    energy_per_mac_nj: float  # energy per MAC in nJ

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
        # assume A: (M,K), B: (K,N) => M*N*K*2 (MACs counted as 1)
        a = shapes[self.A].dims
        b = shapes[self.B].dims
        M, K = a
        K2, N = b
        assert K == K2
        return M * N * K  # MAC FLOPs 估算：这里 MAC 被算成 1 个 flop，而不是 2 个 flop（有的定义会乘以 2）

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
        # rough estimate: each patch flattened multiply with embedding weight
        img_shape = shapes[self.input_img].dims  # (C,H,W)
        C, H, W = img_shape
        n_patches_h = H // self.patch_h
        n_patches_w = W // self.patch_w
        num_patches = n_patches_h * n_patches_w
        patch_size = C * self.patch_h * self.patch_w
        out_dim = shapes[self.weight_name].dims[1]
        return num_patches * patch_size * out_dim   # patching FLOPs 估算：每个 patch 做一次矩阵乘法，patch_size × out_dim 乘法，乘上 patch 数量

    def required_tensors(self) -> List[str]:
        return [self.input_img, self.weight_name]

# ----------------------------- Simple model description -----------------------------

class Model:
    def __init__(self):
        self.tensors: Dict[str, Tensor] = {}
        self.shapes: Dict[str, TensorShape] = {}
        self.ops: List[Op] = []

    def add_tensor(self, name: str, shape: Tuple[int, ...], bytes_per_element=4, device='dram'): # 默认存到dram
        shape_obj = TensorShape(shape)  # 创建一个形状对象，存储维度信息
        size = int(np.prod(shape) * bytes_per_element)  # 总字节数
        self.tensors[name] = Tensor(name, shape_obj, size, device)
        self.shapes[name] = shape_obj

    def add_op(self, op: Op):
        self.ops.append(op) # 追加到算子列表

# ----------------------------- Compiler (mapper + scheduler) -----------------------------

class SimpleCompiler:
    """
    A naive compiler that places weights on RRAM, activations on DRAM (default),
    splits matmuls into tiles that fit into a compute tile size, and produces a schedule.
    """
    def __init__(self, model: Model, rram: MemoryDevice, dram: MemoryDevice, cu: ComputeUnit,
                 bytes_per_element=4, tile_K=256, tile_M=64, tile_N=64):
        self.model = model
        self.rram = rram
        self.dram = dram
        self.cu = cu
        self.tile_K = tile_K
        self.tile_M = tile_M
        self.tile_N = tile_N
        self.bytes_per_element = bytes_per_element

    def place(self):
        # place weights: user may already choose device; otherwise weights declared with name 'W*' go to RRAM
        placements = {}
        for tname, tensor in self.model.tensors.items():
            if tname.startswith('W') or tensor.device == 'rram':
                # attempt allocate in RRAM
                if self.rram.allocate(tensor.size_bytes):
                    placements[tname] = 'rram'
                    tensor.device = 'rram'
                else:
                    placements[tname] = 'dram'
                    tensor.device = 'dram'
            else:
                # activations default to dram
                if self.dram.allocate(tensor.size_bytes):
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
            if isinstance(op, MatMul):
                M, K = self.model.shapes[op.A].dims
                K2, N = self.model.shapes[op.B].dims
                assert K == K2
                # tile over M and N
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
            else:
                schedule.append({'op': op, 'type': 'unknown'})
        return schedule

# ----------------------------- Simulator core -----------------------------

@dataclass
class Stats:
    cycles: int = 0
    energy_nj: float = 0.0
    macs: int = 0
    bytes_read: int = 0
    bytes_written: int = 0

class Simulator:
    def __init__(self, model: Model, schedule: List[Dict[str, Any]], rram: MemoryDevice, dram: MemoryDevice, cu: ComputeUnit, bytes_per_element=4):
        self.model = model
        self.schedule = schedule
        self.rram = rram
        self.dram = dram
        self.cu = cu
        self.stats = Stats()
        self.bpe = bytes_per_element

    def _mem_read_cost(self, dev: MemoryDevice, size_bytes: int):
        # cycles limited by bandwidth + latency
        bw_cycles = math.ceil(size_bytes / dev.read_bw_bytes_per_cycle)
        cycles = dev.access_latency_cycles + bw_cycles
        energy = size_bytes * dev.read_energy_per_byte
        return cycles, energy

    def _mem_write_cost(self, dev: MemoryDevice, size_bytes: int):
        bw_cycles = math.ceil(size_bytes / dev.write_bw_bytes_per_cycle)
        cycles = dev.access_latency_cycles + bw_cycles
        energy = size_bytes * dev.write_energy_per_byte
        return cycles, energy

    def _compute_cost(self, macs: int):
        cycles = math.ceil(macs / self.cu.macs_per_cycle)
        energy = macs * self.cu.energy_per_mac_nj
        return cycles, energy

    def run(self):
        for item in self.schedule:
            op = item['op']
            if item['type'] == 'matmul_tile':
                m = item['msize']; n = item['nsize']; k = item['ksize']
                # sizes in elements
                A_tile_bytes = m * k * self.bpe
                B_tile_bytes = k * n * self.bpe
                # read A
                devA = self.model.tensors[op.A].device
                devB = self.model.tensors[op.B].device
                memA = self.rram if devA=='rram' else self.dram
                memB = self.rram if devB=='rram' else self.dram
                cA, eA = self._mem_read_cost(memA, A_tile_bytes)
                cB, eB = self._mem_read_cost(memB, B_tile_bytes)
                # compute
                macs = m * n * k
                cc, ec = self._compute_cost(macs)
                # write C tile to dram (assume output in dram)
                C_tile_bytes = m * n * self.bpe
                cW, eW = self._mem_write_cost(self.dram, C_tile_bytes)
                # schedule: assume memory reads + compute pipelined; take max of mem and compute
                cycles = max(cA + cB, cc) + cW
                energy = eA + eB + ec + eW
                # update stats
                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.macs += macs
                self.stats.bytes_read += (A_tile_bytes + B_tile_bytes)
                self.stats.bytes_written += C_tile_bytes
            elif item['type'] == 'patch_embed':
                # naive: read image patch and weight (weights likely in rram)
                op: PatchEmbed = op
                img_shape = self.model.shapes[op.input_img].dims
                Cc, H, W = img_shape
                nph = H // op.patch_h
                npw = W // op.patch_w
                num_patches = nph * npw
                patch_size = Cc * op.patch_h * op.patch_w
                out_dim = self.model.shapes[op.weight_name].dims[1]
                # weight read (assume stored in rram)
                weight_bytes = patch_size * out_dim * self.bpe
                cW, eW = self._mem_read_cost(self.rram, weight_bytes)
                # image read (dram)
                img_bytes = self.model.tensors[op.input_img].size_bytes
                cI, eI = self._mem_read_cost(self.dram, img_bytes)
                # compute cost
                macs = num_patches * patch_size * out_dim
                cc, ec = self._compute_cost(macs)
                # write embeddings
                out_bytes = num_patches * out_dim * self.bpe
                cOut, eOut = self._mem_write_cost(self.dram, out_bytes)
                cycles = max(cW + cI, cc) + cOut
                energy = eW + eI + ec + eOut
                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.macs += macs
                self.stats.bytes_read += (weight_bytes + img_bytes)
                self.stats.bytes_written += out_bytes
            else:
                # unknown: skip small cost
                self.stats.cycles += 1
        return self.stats

# ----------------------------- Example usage & default parameters -----------------------------

def build_example_model():
    model = Model()
    # Example: vision patch embedding followed by a small matmul representing attention Q*K
    # image tensor: C=3, H=224, W=224
    model.add_tensor('img', (3,224,224), bytes_per_element=4, device='dram')
    # embedding weight: W_embed: (patch_size, embed_dim)
    patch_h = patch_w = 16
    patch_size = 3 * patch_h * patch_w
    embed_dim = 768
    model.add_tensor('W_embed', (patch_size, embed_dim), bytes_per_element=4, device='rram')
    # after patch embedding: tokens = (num_patches, embed_dim)
    num_patches = (224//16)*(224//16)
    model.add_tensor('tokens', (num_patches, embed_dim), bytes_per_element=4, device='dram')
    # Q,K,V weights stored in RRAM (W_q, W_k, W_v)
    model.add_tensor('W_q', (embed_dim, 256), bytes_per_element=4, device='rram')
    model.add_tensor('W_k', (embed_dim, 256), bytes_per_element=4, device='rram')
    model.add_tensor('W_v', (embed_dim, 256), bytes_per_element=4, device='rram')
    model.add_tensor('Q', (num_patches, 256), bytes_per_element=4, device='dram')
    model.add_tensor('K', (num_patches, 256), bytes_per_element=4, device='dram')
    model.add_tensor('V', (num_patches, 256), bytes_per_element=4, device='dram')

    # ops: patch embedding
    model.add_op(PatchEmbed('img', patch_w, patch_h, 'tokens', 'W_embed'))
    # compute Q = tokens * W_q
    model.add_op(MatMul('tokens', 'W_q', 'Q'))
    model.add_op(MatMul('tokens', 'W_k', 'K'))
    model.add_op(MatMul('tokens', 'W_v', 'V'))
    return model


if __name__ == '__main__':
    # device parameters: these are example values and should be tuned to realistic tech
    rram = MemoryDevice(name='3D_RRAM', capacity_bytes=8*1024*1024, read_bw_bytes_per_cycle=1024, write_bw_bytes_per_cycle=512,
                        read_energy_per_byte=0.0005, write_energy_per_byte=0.0008, access_latency_cycles=10)
    dram = MemoryDevice(name='3D_DRAM', capacity_bytes=64*1024*1024, read_bw_bytes_per_cycle=8192, write_bw_bytes_per_cycle=4096,
                        read_energy_per_byte=0.002, write_energy_per_byte=0.003, access_latency_cycles=50)
    cu = ComputeUnit(macs_per_cycle=4096, energy_per_mac_nj=0.0002)

    model = build_example_model()
    compiler = SimpleCompiler(model, rram, dram, cu, tile_K=256, tile_M=128, tile_N=128)
    schedule = compiler.compile()
    sim = Simulator(model, schedule, rram, dram, cu)

    t0 = time.time()
    stats = sim.run()
    t1 = time.time()

    print("Simulation result (prototype):")
    print(f"Total cycles: {stats.cycles}")
    print(f"Total MACs: {stats.macs}")
    print(f"Total energy (nJ): {stats.energy_nj:.2f}")
    print(f"Bytes read: {stats.bytes_read}")
    print(f"Bytes written: {stats.bytes_written}")
    print(f"Runtime (s): {t1-t0:.3f}")

    # simple derived metrics
    freq_ghz = 1.0  # assume 1 GHz
    exec_time_s = stats.cycles / (freq_ghz * 1e9)
    energy_j = stats.energy_nj * 1e-9
    print(f"Estimated wall time @1GHz: {exec_time_s:.6f} s")
    print(f"Estimated energy (J): {energy_j:.6f} J")

    print('\nNotes:')
    print('- This prototype uses many simplifications: coarse tiling, naive placement, no overlap optimizations.')
    print('- Extend compiler to consider RRAM in-situ compute (if supporting analog MACs), or reduce DRAM traffic by caching.')
    print('- Adjust device parameters to match technology. Energy units: nJ in this proto.')

# End of file
