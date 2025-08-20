# file: simulator.py
import math
import numpy as np
from typing import List, Dict, Any, Tuple

from model import Model
from hardware_models import MemoryDevice, ComputeUnit, Stats
from operations import (
    MatMul, PatchEmbed, LayerNorm,
    SoftmaxOp, GeluOp, ReluOp, SigmoidOp, TanhOp,
    AddOp, SubOp, MulOp, DivOp, UCIeOp, ParallelOp
)


class Simulator:
    def __init__(self, model: Model, schedule: List[Dict[str, Any]], rram: MemoryDevice, dram: MemoryDevice, cu: ComputeUnit, bits_per_element=8):
        self.model = model
        self.schedule = schedule
        self.rram, self.dram = rram, dram
        self.cu = cu
        self.stats = Stats()
        self.bpe_bits = bits_per_element
        self.ucie_bandwidth = 4e12  # bits/cycle  # UCIe PHY for Advanced Packages: 4Tbps (16G*64*4) multi-module PHY @ 16Gbps/pin. 64 TX + 64 RX pins per Module (4 modules)
        self.ucie_energy_per_bit = 0.5  # pJ/bit
        self.layer_latency_max_cycles = 3 # 0.01ns/layer, 256 layer in total
        
    def _layer_delay_cycles(self, layer: int, max_layer: int = 256) -> int:
        if not (1 <= layer <= max_layer) or self.layer_latency_max_cycles <= 0:
            return 0
        # 四舍五入为整数 cycles
        return int(round((layer - 1) / (max_layer - 1) * self.layer_latency_max_cycles))

    def _mem_read_cost(self, dev: MemoryDevice, size_bits: int, src_layer: int = -1):
        bw_cycles = math.ceil(size_bits / dev.read_bw_bits_per_cycle) if dev.read_bw_bits_per_cycle > 0 else 0
        cycles = dev.access_latency_cycles + bw_cycles
        energy = size_bits * dev.read_energy_per_bit

        hops = dev.tsv_hops(src_layer)
        tsv_cycles = dev.tsv_cycles_for(size_bits, hops)
        return cycles + tsv_cycles, energy  # 这里先只加延迟；能量如需可扩展

    def _mem_write_cost(self, dev: MemoryDevice, size_bits: int, src_layer: int = -1):
        bw_cycles = math.ceil(size_bits / dev.write_bw_bits_per_cycle) if dev.write_bw_bits_per_cycle > 0 else 0
        cycles = dev.access_latency_cycles + bw_cycles
        energy = size_bits * dev.write_energy_per_bit

        hops = dev.tsv_hops(src_layer)
        tsv_cycles = dev.tsv_cycles_for(size_bits, hops)

        return cycles + tsv_cycles, energy

    def _compute_cost(self, macs: int):
        cycles = math.ceil(macs / self.cu.macs_per_cycle)
        energy = macs * self.cu.energy_per_mac_nj
        return cycles, energy

    # def _calculate_item_cost(self, item: Dict[str, Any]) -> Tuple[int, float, int, int, int]:
    #     """计算单个 schedule item 的成本，但不更新全局统计数据"""
    #     ttype = item['type']
    #     op = item.get('op')
    #     cycles, energy, macs, bits_read, bits_written = 0, 0, 0, 0, 0

    #     if ttype == 'matmul_tile':
    #         m, n, k = item['msize'], item['nsize'], item['ksize']
    #         A_tile_bits, B_tile_bits, C_tile_bits = m * k * self.bpe_bits, k * n * self.bpe_bits, m * n * self.bpe_bits
    #         devA, devB = self.model.tensors[op.A].device, self.model.tensors[op.B].device
    #         memA, memB = (self.rram if devA == 'rram' else self.dram), (self.rram if devB == 'rram' else self.dram)
    #         cA, eA = self._mem_read_cost(memA, A_tile_bits)
    #         cB, eB = self._mem_read_cost(memB, B_tile_bits)
    #         macs = m * n * k
    #         cc, ec = self._compute_cost(macs)
    #         cW, eW = self._mem_write_cost(self.dram, C_tile_bits)
    #         cycles = max(cA + cB, cc) + cW
    #         energy = eA + eB + ec + eW
    #         bits_read = A_tile_bits + B_tile_bits
    #         bits_written = C_tile_bits

    #     elif ttype == 'patchembed':
    #         op: PatchEmbed = op
    #         Cc, H, W = self.model.shapes[op.input_img].dims
    #         num_patches = (H // op.patch_h) * (W // op.patch_w)
    #         patch_size = Cc * op.patch_h * op.patch_w
    #         out_dim = self.model.shapes[op.weight_name].dims[1]
    #         weight_bits = patch_size * out_dim * self.bpe_bits
    #         cW, eW = self._mem_read_cost(self.rram, weight_bits)
    #         img_bits = self.model.tensors[op.input_img].size_bits
    #         cI, eI = self._mem_read_cost(self.dram, img_bits)
    #         macs = num_patches * patch_size * out_dim
    #         cc, ec = self._compute_cost(macs)
    #         out_bits = num_patches * out_dim * self.bpe_bits
    #         cOut, eOut = self._mem_write_cost(self.dram, out_bits)
    #         cycles = max(cW + cI, cc) + cOut
    #         energy = eW + eI + ec + eOut
    #         bits_read = weight_bits + img_bits
    #         bits_written = out_bits

    #     elif ttype in ('softmaxop', 'geluop', 'reluop', 'sigmoidop', 'tanhop', 'layernorm'):
    #         macs = op.flops(self.model.shapes)
    #         cc, ec = self._compute_cost(macs)
    #         a_bits = self.model.tensors[op.A].size_bits
    #         c_bits = self.model.tensors[op.C].size_bits
    #         rc, re = self._mem_read_cost(self.dram, a_bits)
    #         wc, we = self._mem_write_cost(self.dram, c_bits)
    #         cycles = max(rc, cc) + wc
    #         energy = re + ec + we
    #         bits_read, bits_written = a_bits, c_bits

    #     elif ttype in ('addop', 'subop', 'mulop', 'divop'):
    #         macs = op.flops(self.model.shapes)
    #         cc, ec = self._compute_cost(macs)
    #         a_bits, b_bits, c_bits = self.model.tensors[op.A].size_bits, self.model.tensors[op.B].size_bits, self.model.tensors[op.C].size_bits
    #         rcA, reA = self._mem_read_cost(self.dram, a_bits)
    #         rcB, reB = self._mem_read_cost(self.dram, b_bits)
    #         wc, we = self._mem_write_cost(self.dram, c_bits)
    #         cycles = max(rcA + rcB, cc) + wc
    #         energy = reA + reB + ec + we
    #         bits_read = a_bits + b_bits
    #         bits_written = c_bits

    #     elif ttype == 'ucieop':
    #         op = item['op']
    #         cycles = op.size_bits // self.ucie_bandwidth
    #         energy = op.size_bits * self.ucie_energy_per_bit * 1e-3  # 转 nJ

    #     else:
    #         raise ValueError(f"Unknown ttype '{ttype}' in schedule item: {item}")
        
    #     return cycles, energy, macs, bits_read, bits_written

    def _dev_and_layer(self, tname: str):
        """小工具：由张量名取回 (mem_device, layer, size_bits)"""
        t = self.model.tensors[tname]
        mem = self.rram if t.device == 'rram' else self.dram
        return mem, t.layer, t.size_bits

    def _calculate_item_cost(self, item: Dict[str, Any]) -> Tuple[int, float, int, int, int]:
        ttype = item['type']
        op = item.get('op')
        cycles, energy, macs, bits_read, bits_written = 0, 0, 0, 0, 0

        if ttype == 'matmul_tile':
            m, n, k = item['msize'], item['nsize'], item['ksize']
            A_tile_bits = m * k * self.bpe_bits
            B_tile_bits = k * n * self.bpe_bits
            C_tile_bits = m * n * self.bpe_bits

            # === 读 A、B：用各自 device + layer ===
            memA, layerA, _ = self._dev_and_layer(op.A)
            memB, layerB, _ = self._dev_and_layer(op.B)
            cA, eA = self._mem_read_cost(memA, A_tile_bits, src_layer=layerA)
            cB, eB = self._mem_read_cost(memB, B_tile_bits, src_layer=layerB)

            macs = m * n * k
            cc, ec = self._compute_cost(macs)

            # === 写 C：用 C 的 device + layer（之前硬写 dram 要改）===
            memC, layerC, _ = self._dev_and_layer(op.C)
            cW, eW = self._mem_write_cost(memC, C_tile_bits, src_layer=layerC)

            cycles = max(cA + cB, cc) + cW
            energy = eA + eB + ec + eW
            bits_read = A_tile_bits + B_tile_bits
            bits_written = C_tile_bits

        elif ttype == 'patchembed':
            op: PatchEmbed = op
            Cc, H, W = self.model.shapes[op.input_img].dims
            num_patches = (H // op.patch_h) * (W // op.patch_w)
            patch_size = Cc * op.patch_h * op.patch_w
            out_dim = self.model.shapes[op.weight_name].dims[1]

            # === 权重与输入图像按各自 device+layer 计 TSV ===
            memW, layerW, _ = self._dev_and_layer(op.weight_name)
            weight_bits = patch_size * out_dim * self.bpe_bits
            cWrd, eWrd = self._mem_read_cost(memW, weight_bits, src_layer=layerW)

            memI, layerI, img_bits = self._dev_and_layer(op.input_img)
            cI, eI = self._mem_read_cost(memI, img_bits, src_layer=layerI)

            macs = num_patches * patch_size * out_dim
            cc, ec = self._compute_cost(macs)

            out_bits = num_patches * out_dim * self.bpe_bits
            # === 输出张量按自己 device+layer 写回 ===
            memO, layerO, _ = self._dev_and_layer(op.output_tokens)  # 若你类里叫 op.output_name，就用对应名
            cOut, eOut = self._mem_write_cost(memO, out_bits, src_layer=layerO)

            cycles = max(cWrd + cI, cc) + cOut
            energy = eWrd + eI + ec + eOut
            bits_read = weight_bits + img_bits
            bits_written = out_bits

        elif ttype in ('softmaxop', 'geluop', 'reluop', 'sigmoidop', 'tanhop', 'layernorm'):
            macs = op.flops(self.model.shapes)
            cc, ec = self._compute_cost(macs)

            # === A 的读、C 的写都带 device+layer ===
            memA, layerA, a_bits = self._dev_and_layer(op.A)
            memC, layerC, c_bits = self._dev_and_layer(op.C)
            rc, re = self._mem_read_cost(memA, a_bits, src_layer=layerA)
            wc, we = self._mem_write_cost(memC, c_bits, src_layer=layerC)

            cycles = max(rc, cc) + wc
            energy = re + ec + we
            bits_read, bits_written = a_bits, c_bits

        elif ttype in ('addop', 'subop', 'mulop', 'divop'):
            macs = op.flops(self.model.shapes)
            cc, ec = self._compute_cost(macs)

            memA, layerA, a_bits = self._dev_and_layer(op.A)
            memB, layerB, b_bits = self._dev_and_layer(op.B)
            memC, layerC, c_bits = self._dev_and_layer(op.C)

            rcA, reA = self._mem_read_cost(memA, a_bits, src_layer=layerA)
            rcB, reB = self._mem_read_cost(memB, b_bits, src_layer=layerB)
            wc, we  = self._mem_write_cost(memC, c_bits, src_layer=layerC)

            cycles = max(rcA + rcB, cc) + wc
            energy = reA + reB + ec + we
            bits_read = a_bits + b_bits
            bits_written = c_bits

        elif ttype == 'ucieop':
            op = item['op']
            cycles = op.size_bits // self.ucie_bandwidth
            energy = op.size_bits * self.ucie_energy_per_bit * 1e-3  # nJ

        else:
            raise ValueError(f"Unknown ttype '{ttype}' in schedule item: {item}")

        return cycles, energy, macs, bits_read, bits_written


    def _simulate_parallel(self, parallel_item: Dict[str, Any]):
        branch_stats = []
        for branch_schedule in parallel_item['branches']:
            b_cycles, b_energy, b_macs, b_br, b_bw = 0, 0, 0, 0, 0
            for item_in_branch in branch_schedule:
                c, e, m, br, bw = self._calculate_item_cost(item_in_branch)
                b_cycles += c
                b_energy += e
                b_macs += m
                b_br += br
                b_bw += bw
            branch_stats.append((b_cycles, b_energy, b_macs, b_br, b_bw))
        
        # 并行执行时间取决于最长的分支，而能耗、计算量等是所有分支的总和
        cycles = max(s[0] for s in branch_stats) if branch_stats else 0
        energy = sum(s[1] for s in branch_stats)
        macs = sum(s[2] for s in branch_stats)
        bits_read = sum(s[3] for s in branch_stats)
        bits_written = sum(s[4] for s in branch_stats)

        # 更新全局统计数据
        self.stats.cycles += cycles
        self.stats.energy_nj += energy
        self.stats.macs += macs
        self.stats.bits_read += bits_read
        self.stats.bits_written += bits_written
        self.stats.breakdown['parallel'] = self.stats.breakdown.get('parallel', 0) + energy
        self.stats.cycles_breakdown['parallel'] = self.stats.cycles_breakdown.get('parallel', 0) + cycles

    def run(self):
        for item in self.schedule:
            ttype = item['type']
            if ttype == 'parallel':
                self._simulate_parallel(item)
            else:
                # 串行执行单个任务
                cycles, energy, macs, bits_read, bits_written = self._calculate_item_cost(item)
                
                # 更新全局统计数据
                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.macs += macs
                self.stats.bits_read += bits_read
                self.stats.bits_written += bits_written
                self.stats.breakdown[ttype] = self.stats.breakdown.get(ttype, 0) + energy
                self.stats.macs_breakdown[ttype] = self.stats.macs_breakdown.get(ttype, 0) + macs
                self.stats.cycles_breakdown[ttype] = self.stats.cycles_breakdown.get(ttype, 0) + cycles
        
        return self.stats