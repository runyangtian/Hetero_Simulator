# file: simulator.py
import math
import numpy as np
from typing import List, Dict, Any

from model import Model
from hardware_models import MemoryDevice, ComputeUnit, ACU, Stats
from operations import (
    MatMul, PatchEmbed, LayerNorm,
    SoftmaxOp, GeluOp, ReluOp, SigmoidOp, TanhOp,
    AddOp, SubOp, MulOp, DivOp, UCIeOp
)


class Simulator:
    def __init__(self, model: Model, schedule: List[Dict[str, Any]], rram: MemoryDevice, dram: MemoryDevice, cu: ComputeUnit, bits_per_element=32):
        self.model = model
        self.schedule = schedule
        self.rram, self.dram = rram, dram
        self.cu = cu
        self.stats = Stats()
        self.bpe_bits = bits_per_element
        self.ucie_bandwidth = 64  # bits/cycle
        self.ucie_energy_per_bit = 0.5  # pJ/bit


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
                m = item['msize']
                n = item['nsize']
                k = item['ksize']
                # tile 数据大小（bit）
                A_tile_bits = m * k * self.bpe_bits
                B_tile_bits = k * n * self.bpe_bits
                C_tile_bits = m * n * self.bpe_bits
                # memory device
                devA = self.model.tensors[op.A].device
                devB = self.model.tensors[op.B].device
                memA = self.rram if devA == 'rram' else self.dram
                memB = self.rram if devB == 'rram' else self.dram
                # memory read cost
                cA, eA = self._mem_read_cost(memA, A_tile_bits)
                cB, eB = self._mem_read_cost(memB, B_tile_bits)
                # compute cost
                macs = m * n * k
                cc, ec = self._compute_cost(macs)
                # memory write cost (写回 DRAM)
                cW, eW = self._mem_write_cost(self.dram, C_tile_bits)
                cycles = max(cA + cB, cc) + cW
                energy = eA + eB + ec + eW  # read A, read B, write Result, computing

                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.macs += macs
                self.stats.bits_read += (A_tile_bits + B_tile_bits)
                self.stats.bits_written += C_tile_bits
                self.stats.breakdown['matmul'] = self.stats.breakdown.get('matmul', 0) + energy

            elif ttype == 'patchembed':
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
                self.stats.breakdown['patch_embed'] = self.stats.breakdown.get('patch_embed',0) + energy

            # 单输入/双输入：这么建模太粗糙，需要修改

            elif ttype in ('softmaxop', 'geluop', 'reluop', 'sigmoidop', 'tanhop', 'layernorm'):    # 单输入算子
                macs = op.flops(self.model.shapes)
                cc, ec = self._compute_cost(macs)
                a_bits = self.model.tensors[op.A].size_bits
                c_bits = self.model.tensors[op.C].size_bits
                rc, re = self._mem_read_cost(self.dram, a_bits)
                wc, we = self._mem_write_cost(self.dram, c_bits)
                cycles = max(rc, cc) + wc
                energy = re + ec + we

                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.bits_read += a_bits
                self.stats.bits_written += c_bits
                self.stats.breakdown[ttype] = self.stats.breakdown.get(ttype, 0) + energy

            elif ttype in ('addop', 'subop', 'mulop', 'divop'):     # 双输入算子
                macs = op.flops(self.model.shapes)
                cc, ec = self._compute_cost(macs)
                a_bits = self.model.tensors[op.A].size_bits
                b_bits = self.model.tensors[op.B].size_bits
                c_bits = self.model.tensors[op.C].size_bits
                rcA, reA = self._mem_read_cost(self.dram, a_bits)
                rcB, reB = self._mem_read_cost(self.dram, b_bits)
                wc, we = self._mem_write_cost(self.dram, c_bits)
                cycles = max(rcA + rcB, cc) + wc
                energy = reA + reB + ec + we

                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.bits_read += (a_bits + b_bits)
                self.stats.bits_written += c_bits
                self.stats.breakdown[ttype] = self.stats.breakdown.get(ttype, 0) + energy

            elif ttype == 'ucieop':
                op = item['op']
                BW_bits_per_cycle = self.ucie_bandwidth
                energy_per_bit = self.ucie_energy_per_bit

                cycles = op.size_bits // BW_bits_per_cycle
                energy = op.size_bits * energy_per_bit * 1e-3  # 转 nJ

                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.breakdown[ttype] = self.stats.breakdown.get(ttype, 0) + energy


            else:
                raise ValueError(f"Unknown ttype '{ttype}' in schedule item: {item}")

        return self.stats