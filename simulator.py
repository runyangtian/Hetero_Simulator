# file: simulator.py
import math
import numpy as np
from typing import List, Dict, Any, Tuple
from model import Model
from hardware_models import MemoryDevice, ComputeUnit, Stats
from operations import (
    MatMul, Conv2D, AvgPool2D, LayerNorm,
    SoftmaxOp, GeluOp, ReluOp, SigmoidOp, TanhOp,
    AddOp, SubOp, MulOp, DivOp, UCIeOp, ParallelOp
)


class Simulator:
    def __init__(self, model: Model, schedule: List[Dict[str, Any]], rram: MemoryDevice, dram: MemoryDevice, dram_cu: ComputeUnit, rram_cu: ComputeUnit):
        self.model = model
        self.schedule = schedule
        self.rram, self.dram = rram, dram

        # self.cu = cu
        self.dram_cu = dram_cu
        self.rram_cu = rram_cu

        self.stats = Stats()
        # self.bpe_bits = bits_per_element
        self.ucie_bandwidth = 2048      # 32 Gb/s × 64 = 2,048 Gb/s；1 GHz → 2,048 Gb/s ÷ 1e9 = 2,048 bit/cycle，全双工则*2
        self.ucie_energy_per_bit = 0.5  # pJ/bit
        self.layer_latency_max_cycles = 3 # 0.01ns/layer, 256 layer in total


    def _mem_read_cost(self, dev: MemoryDevice, size_bits: int, src_layer: int = -1):
        bw_cycles = math.ceil(size_bits / dev.read_bw_bits_per_cycle) if dev.read_bw_bits_per_cycle > 0 else 0
        cycles = dev.read_latency_cycles + bw_cycles
        energy = size_bits * dev.read_energy_per_bit

        hops = dev.tsv_hops(src_layer)
        tsv_cycles = dev.tsv_cycles_for(size_bits, hops)
        if tsv_cycles == 0:
            return 0, 0
        else:
            return cycles + tsv_cycles, energy  # 这里先只加延迟；能量如需可扩展

    def _mem_write_cost(self, dev: MemoryDevice, size_bits: int, src_layer: int = -1):
        bw_cycles = math.ceil(size_bits / dev.write_bw_bits_per_cycle) if dev.write_bw_bits_per_cycle > 0 else 0
        cycles = dev.write_latency_cycles + bw_cycles
        energy = size_bits * dev.write_energy_per_bit

        hops = dev.tsv_hops(src_layer)
        tsv_cycles = dev.tsv_cycles_for(size_bits, hops)
        if tsv_cycles == 0:
            return 0, 0
        else:
            return cycles + tsv_cycles, energy

    def _compute_cost_engine(self, amount_ops: int, engine: str, cu: ComputeUnit):
        if amount_ops <= 0:
            return 0, 0.0

        if engine == 'sfe':
            tput = cu.sfe_ops_per_cycle
            cycles = (amount_ops + tput - 1) // tput
            energy = amount_ops * cu.sfe_energy_per_op_nj
        else:  # 'mac'``
            tput = max(1, cu.macs_per_cycle)
            cycles = (amount_ops + tput - 1) // tput
            energy = amount_ops * cu.energy_per_mac_nj

        return cycles, energy

    def _dev_and_layer(self, tname: str):
        """小工具：由张量名取回 (mem_device, layer, size_bits)"""
        t = self.model.tensors[tname]
        mem = self.rram if t.device == 'rram' else self.dram
        return mem, t.layer, t.bits_per_element

    def _calculate_item_cost(self, item: Dict[str, Any]) -> Tuple[int, float, int, int, int]:
        ttype = item['type']
        op = item.get('op')
        cycles, energy, macs, bits_read, bits_written = 0, 0, 0, 0, 0

        if ttype == 'matmul_tile':
            m, n, k = item['msize'], item['nsize'], item['ksize']

            # --- 设备与层 ---
            memA, layerA, bpeA = self._dev_and_layer(op.A)
            memB, layerB, bpeB = self._dev_and_layer(op.B)
            memC, layerC, bpeC = self._dev_and_layer(op.C)

            # print(bpeA, bpeB, bpeC)
            A_tile_bits = m * k * bpeA
            B_tile_bits = k * n * bpeB
            C_tile_bits = m * n * bpeC

            # --- 读 A/B（与你原先一致）---
            cA, eA = self._mem_read_cost(memA, A_tile_bits, src_layer=layerA)
            cB, eB = self._mem_read_cost(memB, B_tile_bits, src_layer=layerB)
            c_in = max(cA, cB)

            # --- 选择 CU：按 C 的 device（你之前的逻辑）---
            use_rram = (memC is self.rram)
            cu_sel = self.rram_cu if use_rram else self.dram_cu

            # --- 计算吞吐（近似）---
            macs = m * n * k
            # 用你的引擎计算能量，但忽略它返回的周期（我们自己算周期）
            _, ec = self._compute_cost_engine(macs, 'mac', cu_sel)

            # 计算周期 ≈ 吞吐周期 + 一条波的填/排空
            cc_tput = (macs + cu_sel.macs_per_cycle - 1) // cu_sel.macs_per_cycle
            cc = cc_tput #+ wave_ovhd

            # --- 写回 C ---
            cW, eW = self._mem_write_cost(memC, C_tile_bits, src_layer=layerC)

            # --- 汇总：读与算重叠，最后再写 ---
            # cycles = c_in + cc + cW
            # print(c_in, cc, cW)
            
            cycles = max(c_in, cc, cW)
            energy = eA + eB + ec + eW

            bits_read = A_tile_bits + B_tile_bits
            bits_written = C_tile_bits

        elif ttype == 'conv2d':
            op: Conv2D = op
            C_in, H, W = self.model.shapes[op.input_img].dims
            wshape = self.model.shapes[op.weight_name].dims  # [Cpatch, Cout] 或 [Cout, Cin/groups, Kh, Kw]

            # === 读权重 ===
            memW, layerW, weight_bits = self._dev_and_layer(op.weight_name)
            cWrd, eWrd = self._mem_read_cost(memW, weight_bits, src_layer=layerW)

            # === 读输入 ===
            memI, layerI, img_bits = self._dev_and_layer(op.input_img)
            cI, eI = self._mem_read_cost(memI, img_bits, src_layer=layerI)

            # === 输出空间尺寸 ===
            Ho = (H + 2*op.ph - op.kh) // op.sh + 1
            Wo = (W + 2*op.pw - op.kw) // op.sw + 1

            memO, layerO, bpeO = self._dev_and_layer(op.out_name)
            # === GEMM 等价尺寸 + MACs / out_bits ===
            if len(wshape) == 2:
                # Patch-Embed: [Cpatch, Cout]
                Cpatch, Cout = wshape[0], wshape[1]
                m = Ho * Wo
                k = Cpatch
                n = Cout
                macs = m * k * n
                out_bits = m * n * bpeO
            else:
                # 通用卷积: [Cout, Cin/groups, Kh, Kw]
                Cout = wshape[0]
                Cin_per_g = C_in // max(1, op.groups)
                m = Ho * Wo
                k = Cin_per_g * op.kh * op.kw
                n = Cout
                macs = m * k * n
                out_bits = m * n * bpeO

            # === 选择 CU：按输出张量所在 device（与你原逻辑保持一致）===
            
            cu_sel = self.rram_cu if (memO is self.rram) else self.dram_cu

            # 吞吐周期（近似）+ 一条波填/排空
            cc_tput = (macs + cu_sel.macs_per_cycle - 1) // cu_sel.macs_per_cycle
            # wave_ovhd = self._tile_wave_overhead(cu_sel, m, n)
            cc = cc_tput #+ wave_ovhd

            # 计算能量（保持你原接口；周期我们用上面的 cc）
            _, ec = self._compute_cost_engine(macs, 'mac', cu_sel)

            # === 写回输出 ===
            cOut, eOut = self._mem_write_cost(memO, out_bits, src_layer=layerO)

            
            # === 汇总（读与算重叠，最后写）===
            cycles = max(max(cWrd, cI), cc, cOut)
            # cycles = max(cWrd, cI) + cc + cOut
            energy = eWrd + eI + ec + eOut
            bits_read = weight_bits + img_bits
            bits_written = out_bits

        elif ttype == 'avgpool2d':
            op: AvgPool2D = op
            Cc, H, W = self.model.shapes[op.input_img].dims

            # 输入读取
            memI, layerI, img_bits = self._dev_and_layer(op.input_img)
            cI, eI = self._mem_read_cost(memI, img_bits, src_layer=layerI)

            # 输出尺寸
            Ho = (H + 2*op.ph - op.kh) // op.sh + 1
            Wo = (W + 2*op.pw - op.kw) // op.sw + 1

            # 近似计算量（交给 SFE/special function engine）
            macs = Cc * Ho * Wo * op.kh * op.kw
            memO, layerO, bpeO = self._dev_and_layer(op.out_name)
            cu_sel = self.dram_cu  # 池化通常走 DRAM 侧 SFE
            cc, ec = self._compute_cost_engine(macs, 'sfe', cu_sel)

            # 写回
            out_bits = Cc * Ho * Wo * bpeO
            cOut, eOut = self._mem_write_cost(memO, out_bits, src_layer=layerO)
            cycles = cI + cc + cOut
            energy = eI + ec + eOut
            bits_read, bits_written = img_bits, out_bits

        elif ttype in ('softmaxop', 'geluop', 'reluop', 'sigmoidop', 'tanhop', 'layernorm'):
            macs = op.flops(self.model.shapes)
            # === A 的读、C 的写都带 device+layer ===
            memA, layerA, a_bits = self._dev_and_layer(op.A)
            memC, layerC, c_bits = self._dev_and_layer(op.C)
            cu_sel = self.rram_cu if (memC is self.rram) else self.dram_cu
            cc, ec = self._compute_cost_engine(macs, 'sfe', cu_sel)
            rc, re = self._mem_read_cost(memA, a_bits, src_layer=layerA)
            wc, we = self._mem_write_cost(memC, c_bits, src_layer=layerC)

            cycles = max(rc, cc) + wc
            energy = re + ec + we
            bits_read, bits_written = a_bits, c_bits

        elif ttype in ('addop', 'subop', 'mulop', 'divop'):
            macs = op.flops(self.model.shapes)
            # cc, ec = self._compute_cost(macs)
            cc, ec = self._compute_cost_engine(macs, 'mac', self.dram_cu)

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
        branch_stats = []  # 每项: (b_cycles, b_energy, b_macs, b_br, b_bw, b_cycles_by_type)

        for branch_schedule in parallel_item['branches']:
            b_cycles = b_energy = b_macs = b_br = b_bw = 0
            # 记录该分支里“按算子类型”的 cycles 分布
            b_cycles_by_type: Dict[str, int] = {}

            for item_in_branch in branch_schedule:
                # 若分支里还有 parallel：递归
                if item_in_branch.get('type') == 'parallel':
                    c2, e2, m2, br2, bw2, cbt2 = self._simulate_parallel(item_in_branch)
                    # 递归调用内部已更新全局的能耗/MAC breakdown；这里只做分支累计
                    b_cycles += c2
                    b_energy += e2
                    b_macs   += m2
                    b_br     += br2
                    b_bw     += bw2
                    # 合并子并行返回的“按类型 cycles”
                    for t, cpart in cbt2.items():
                        b_cycles_by_type[t] = b_cycles_by_type.get(t, 0) + cpart
                    continue

                # 普通算子
                c, e, m, br, bw = self._calculate_item_cost(item_in_branch)
                b_cycles += c
                b_energy += e
                b_macs   += m
                b_br     += br
                b_bw     += bw

                # —— 能耗 / MAC 直接归到算子类型（不产生 “parallel” 桶）——
                t = item_in_branch['type']
                self.stats.breakdown[t]      = self.stats.breakdown.get(t, 0) + e
                self.stats.macs_breakdown[t] = self.stats.macs_breakdown.get(t, 0) + m

                # —— 为本分支累计“按类型 cycles”，稍后若该分支成为关键路径则并入全局 —— 
                b_cycles_by_type[t] = b_cycles_by_type.get(t, 0) + c

            branch_stats.append((b_cycles, b_energy, b_macs, b_br, b_bw, b_cycles_by_type))

        # 若没有分支
        if not branch_stats:
            return 0, 0, 0, 0, 0, {}

        # 关键路径（cycles 最大的分支）
        winner = max(branch_stats, key=lambda x: x[0])

        # 并行块：总时间取最长分支；能量/MAC/流量为分支求和
        cycles       = winner[0]
        energy       = sum(s[1] for s in branch_stats)
        macs         = sum(s[2] for s in branch_stats)
        bits_read    = sum(s[3] for s in branch_stats)
        bits_written = sum(s[4] for s in branch_stats)

        # —— 更新全局总量（不写 parallel 桶）——
        self.stats.cycles       += cycles
        self.stats.energy_nj    += energy
        self.stats.macs         += macs
        self.stats.bits_read    += bits_read
        self.stats.bits_written += bits_written

        # —— 只把关键路径分支的 cycles_by_type 计入全局 cycles_breakdown —— 
        winner_cycles_by_type = winner[5]
        for t, cpart in winner_cycles_by_type.items():
            self.stats.cycles_breakdown[t] = self.stats.cycles_breakdown.get(t, 0) + cpart

        # 将关键路径的 cycles_by_type 返回给上层（用于父级并行合并）
        return cycles, energy, macs, bits_read, bits_written, winner_cycles_by_type

    def run(self):
        for item in self.schedule:
            ttype = item['type']
            if ttype == 'parallel':
                # 返回值用于嵌套时向上层传递；本层不再重复累计
                _ = self._simulate_parallel(item)
            else:
                cycles, energy, macs, bits_read, bits_written = self._calculate_item_cost(item)

                # 更新全局总量
                self.stats.cycles       += cycles
                self.stats.energy_nj    += energy
                self.stats.macs         += macs
                self.stats.bits_read    += bits_read
                self.stats.bits_written += bits_written

                # breakdown：能耗/MAC/周期都按算子类型记
                self.stats.breakdown[ttype]        = self.stats.breakdown.get(ttype, 0) + energy
                self.stats.macs_breakdown[ttype]   = self.stats.macs_breakdown.get(ttype, 0) + macs
                self.stats.cycles_breakdown[ttype] = self.stats.cycles_breakdown.get(ttype, 0) + cycles

        return self.stats
