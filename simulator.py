# file: simulator.py
import math
import numpy as np
from typing import List, Dict, Any

from model import Model
from hardware_models import MemoryDevice, ComputeUnit, ACU, Stats
from operations import (
    Op, MatMul, PatchEmbed, LayerNorm, Attention, UnaryOp, BinaryOp, SoftmaxOp
)


class Simulator:
    def __init__(self, model: Model, schedule: List[Dict[str, Any]], rram: MemoryDevice, dram: MemoryDevice, cu: ComputeUnit, acu: ACU, bits_per_element=32):
        self.model = model
        self.schedule = schedule
        self.rram, self.dram = rram, dram
        self.cu, self.acu = cu, acu
        self.stats = Stats()
        self.bpe_bits = bits_per_element

    def _mem_read_cost(self, dev: MemoryDevice, size_bits: int):
        bw_cycles = math.ceil(size_bits / dev.read_bw_bits_per_cycle) if dev.read_bw_bits_per_cycle > 0 else 0
        cycles = dev.access_latency_cycles + bw_cycles
        energy = size_bits * dev.read_energy_per_bit
        return cycles, energy

    def _mem_write_cost(self, dev: MemoryDevice, size_bits: int):
        bw_cycles = math.ceil(size_bits / dev.write_bw_bits_per_cycle) if dev.write_bw_bits_per_cycle > 0 else 0
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
                    L, d = Q_shape
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
