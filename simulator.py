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
        self.dram_cu = dram_cu
        self.rram_cu = rram_cu
        self.stats = Stats()
        self.ucie_bandwidth = 2048      # 32 Gb/s × 64 = 2,048 Gb/s；1 GHz → 2,048 Gb/s ÷ 1e9 = 2,048 bit/cycle，全双工则*2 
        # self.ucie_bandwidth = float("inf")
        self.ucie_energy_per_bit = 0.5  # pJ/bit
        # self.ucie_energy_per_bit = 0
        self.layer_latency_max_cycles = 3 # 0.01ns/layer, 256 layer in total
    
    # accumulate read/write energy and cycle
    def _acc_rw(self, dev: MemoryDevice, cycles: int, energy_nj: float, is_read: bool):
        if dev is self.dram:
            if is_read:
                self.stats.cycles_read_dram += cycles
                self.stats.energy_read_dram_nj += energy_nj
            else:
                self.stats.cycles_write_dram += cycles
                self.stats.energy_write_dram_nj += energy_nj
        elif dev is self.rram:
            if is_read:
                self.stats.cycles_read_rram += cycles
                self.stats.energy_read_rram_nj += energy_nj
            else:
                self.stats.cycles_write_rram += cycles
                self.stats.energy_write_rram_nj += energy_nj
    # accumulate computation energy and cycle
    def _acc_comp(self, cu: ComputeUnit, cycles: int, energy_nj: float):
        if cu is self.dram_cu:
            self.stats.cycles_comp_dram += cycles
            self.stats.energy_comp_dram_nj += energy_nj
        elif cu is self.rram_cu:
            self.stats.cycles_comp_rram += cycles
            self.stats.energy_comp_rram_nj += energy_nj

    # calculate mem read cost
    def _mem_read_cost(self, dev: MemoryDevice, size_bits: int, src_layer: int = -1):
        bw_cycles = math.ceil(size_bits / dev.read_bw_bits_per_cycle) if dev.read_bw_bits_per_cycle > 0 else 0
        cycles = dev.read_latency_cycles + bw_cycles
        energy = size_bits * dev.read_energy_per_bit

        hops = dev.tsv_hops(src_layer)
        tsv_cycles = dev.tsv_cycles_for(size_bits, hops)
        if hops==0:
            return 0, 0.0
        else:
            return cycles + tsv_cycles, energy

    # calculate mem write cost
    def _mem_write_cost(self, dev: MemoryDevice, size_bits: int, src_layer: int = -1):
        bw_cycles = math.ceil(size_bits / dev.write_bw_bits_per_cycle) if dev.write_bw_bits_per_cycle > 0 else 0
        cycles = dev.write_latency_cycles + bw_cycles
        energy = size_bits * dev.write_energy_per_bit

        hops = dev.tsv_hops(src_layer)
        tsv_cycles = dev.tsv_cycles_for(size_bits, hops)
        if hops==0:
            return 0, 0.0
        else:
            return cycles + tsv_cycles, energy

    # calculate computation cost
    def _compute_cost_engine(self, amount_ops: int, engine: str, cu: ComputeUnit):
        if amount_ops <= 0:
            return 0, 0.0

        if engine == 'sfe':
            cycles = (amount_ops + cu.sfe_ops_per_cycle - 1) // cu.sfe_ops_per_cycle
            energy = amount_ops * cu.sfe_energy_per_op_nj
        else:  # 'mac'
            cycles = (amount_ops + cu.macs_per_cycle - 1) // cu.macs_per_cycle
            energy = amount_ops * cu.energy_per_mac_nj

        return cycles, energy

    # get mem_device, layer, size_bits
    def _dev_and_layer(self, tname: str):
        t = self.model.tensors[tname]
        mem = self.rram if t.device == 'rram' else self.dram
        return mem, t.layer, t.bits_per_element

    def _calculate_item_cost(self, item: Dict[str, Any]) -> Tuple[int, float, int, int, int]:
        ttype = item['type']
        op = item.get('op')
        cycles, energy, macs, bits_read, bits_written = 0, 0, 0, 0, 0

        if ttype == 'matmul_tile':
            m, n, k = item['msize'], item['nsize'], item['ksize']

            memA, layerA, bpeA = self._dev_and_layer(op.A)
            memB, layerB, bpeB = self._dev_and_layer(op.B)
            memC, layerC, bpeC = self._dev_and_layer(op.C)

            # print(bpeA, bpeB, bpeC)
            A_tile_bits = m * k * bpeA
            B_tile_bits = k * n * bpeB
            C_tile_bits = m * n * bpeC

            # --- read A/B---
            cA, eA = self._mem_read_cost(memA, A_tile_bits, src_layer=layerA)
            cB, eB = self._mem_read_cost(memB, B_tile_bits, src_layer=layerB)
            c_in = max(cA, cB)

            # --- select CU：based on C's device---
            use_rram = (memC is self.rram)
            cu_sel = self.rram_cu if use_rram else self.dram_cua


            macs = m * n * k
            cc, ec = self._compute_cost_engine(macs, 'mac', cu_sel)
            cW, eW = self._mem_write_cost(memC, C_tile_bits, src_layer=layerC)

            self._acc_rw(memA, c_in, eA, is_read=True)
            self._acc_rw(memB, 0, eB, is_read=True)
            self._acc_comp(cu_sel, cc, ec)
            self._acc_rw(memC, cW, eW, is_read=False)
            
            cycles = max(c_in, cc, cW)
            energy = eA + eB + ec + eW

            bits_read = A_tile_bits + B_tile_bits
            bits_written = C_tile_bits

        elif ttype == 'conv2d':
            op: Conv2D = op
            C_in, H, W = self.model.shapes[op.input_img].dims
            wshape = self.model.shapes[op.weight_name].dims  # [Cpatch, Cout] 或 [Cout, Cin/groups, Kh, Kw]

            # === read input ===
            memI, layerI, bpeI = self._dev_and_layer(op.input_img)
            memO, layerO, bpeO = self._dev_and_layer(op.out_name)

            img_bits_total = C_in * H * W * bpeI
            cI, eI = self._mem_read_cost(memI, img_bits_total, src_layer=layerI)

            # === read weights ===
            memW, layerW, bpeW = self._dev_and_layer(op.weight_name)

            # === height and width ===
            Ho = (H + 2*op.ph - op.kh) // op.sh + 1
            Wo = (W + 2*op.pw - op.kw) // op.sw + 1

            # === GEMM 等价尺寸 + MACs / out_bits ===
            if len(wshape) == 2:
                # Patch-Embed: [Cpatch, Cout]
                Cpatch, Cout = wshape[0], wshape[1]
                weight_bits_total = Cpatch * Cout * bpeW
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
                weight_bits_total = Cout * k * bpeW
                n = Cout
                macs = m * k * n
                out_bits = m * n * bpeO

            cWrd, eWrd = self._mem_read_cost(memW, weight_bits_total, src_layer=layerW)
            c_in = max(cWrd, cI) 
            cu_sel = self.rram_cu if (memO is self.rram) else self.dram_cu
            cc, ec = self._compute_cost_engine(macs, 'mac', cu_sel)
            cOut, eOut = self._mem_write_cost(memO, out_bits, src_layer=layerO)
            
            # print(c_in, cc, cOut)
            self._acc_rw(memI, c_in, eI, is_read=True)
            self._acc_rw(memW, 0, eWrd, is_read=True)
            self._acc_comp(cu_sel, cc, ec)
            self._acc_rw(memO, cOut, eOut, is_read=False)
            
            # === 汇总（读与算重叠，最后写）===
            cycles = max(c_in, cc, cOut)
            energy = eWrd + eI + ec + eOut
            bits_read = weight_bits_total + img_bits_total
            bits_written = out_bits

        elif ttype == 'avgpool2d':
            op: AvgPool2D = op
            Cc, H, W = self.model.shapes[op.input_img].dims

            # 输入读取
            memI, layerI, bpeI = self._dev_and_layer(op.input_img)
            img_bits_total = Cc * H * W * bpeI
            cI, eI = self._mem_read_cost(memI, img_bits_total, src_layer=layerI)

            # 输出尺寸
            Ho = (H + 2*op.ph - op.kh) // op.sh + 1
            Wo = (W + 2*op.pw - op.kw) // op.sw + 1
            macs = Cc * Ho * Wo * op.kh * op.kw

            memO, layerO, bpeO = self._dev_and_layer(op.out_name)
            cu_sel = self.rram_cu if (memO is self.rram) else self.dram_cu
            cc, ec = self._compute_cost_engine(macs, 'sfe', cu_sel)

            # 写回
            out_bits = Cc * Ho * Wo * bpeO
            cOut, eOut = self._mem_write_cost(memO, out_bits, src_layer=layerO)

            self._acc_rw(memI, cI, eI, is_read=True)
            self._acc_comp(cu_sel, cc, ec)
            self._acc_rw(memO, cOut, eOut, is_read=False)
            cycles = max(cI, cc, cOut)
            energy = eI + ec + eOut
            bits_read, bits_written = img_bits_total, out_bits

        elif ttype in ('softmaxop', 'geluop', 'reluop', 'sigmoidop', 'tanhop', 'layernorm'):
            macs = op.flops(self.model.shapes)
            # === A 的读、C 的写都带 device+layer ===
            memA, layerA, bpeA = self._dev_and_layer(op.A)
            memC, layerC, bpeC = self._dev_and_layer(op.C)
           
            wshapeA = self.model.shapes[op.A].dims
            wshapeC = self.model.shapes[op.C].dims

            if len(wshapeA)==2:
                a_bits_total = wshapeA[0] * wshapeA[1] * bpeA 
            elif len(wshapeA)==3:
                a_bits_total = wshapeA[0] * wshapeA[1] * wshapeA[2] * bpeA
            else:
                raise ValueError(f"fix the code in simulator for A, dims={wshapeA}")

            if len(wshapeC)==2:
                c_bits_total = wshapeC[0] * wshapeC[1] * bpeC
            elif len(wshapeC)==3: 
                c_bits_total = wshapeC[0] * wshapeC[1] * wshapeC[2] * bpeC
            else:
                raise ValueError(f"fix the code in simulator for C, dims={wshapeC}")

            cu_sel = self.rram_cu if (memC is self.rram) else self.dram_cu
            cc, ec = self._compute_cost_engine(macs, 'sfe', cu_sel)
            rc, re = self._mem_read_cost(memA, a_bits_total, src_layer=layerA)
            wc, we = self._mem_write_cost(memC, c_bits_total, src_layer=layerC)

            self._acc_rw(memA, rc, re, is_read=True)
            self._acc_comp(cu_sel, cc, ec)
            self._acc_rw(memC, wc, we, is_read=False)

            cycles = max(rc, cc, wc)
            energy = re + ec + we
            bits_read, bits_written = a_bits_total, c_bits_total

        elif ttype in ('addop', 'subop', 'mulop', 'divop'):
            macs = op.flops(self.model.shapes)

            memA, layerA, bpeA = self._dev_and_layer(op.A)
            memB, layerB, bpeB = self._dev_and_layer(op.B)
            memC, layerC, bpeC = self._dev_and_layer(op.C)

            wshapeA = self.model.shapes[op.A].dims
            wshapeB = self.model.shapes[op.B].dims
            wshapeC = self.model.shapes[op.C].dims

            if len(wshapeA) == 2:
                a_bits_total = wshapeA[0] * wshapeA[1] * bpeA
            elif len(wshapeA) == 3:
                a_bits_total = wshapeA[0] * wshapeA[1] * wshapeA[2] * bpeA
            else:
                raise ValueError(f"fix the code in simulator for A, dims={wshapeA}")

            if len(wshapeB) == 2:
                b_bits_total = wshapeB[0] * wshapeB[1] * bpeB
            elif len(wshapeB) == 3:
                b_bits_total = wshapeB[0] * wshapeB[1] * wshapeB[2] * bpeB
            else:
                raise ValueError(f"fix the code in simulator for B, dims={wshapeB}")

            if len(wshapeC) == 2:
                c_bits_total = wshapeC[0] * wshapeC[1] * bpeC
            elif len(wshapeC) == 3:
                c_bits_total = wshapeC[0] * wshapeC[1] * wshapeC[2] * bpeC
            else:
                raise ValueError(f"fix the code in simulator for C, dims={wshapeC}")

            cu_sel = self.rram_cu if (memC is self.rram) else self.dram_cu
            cc, ec = self._compute_cost_engine(macs, 'mac', cu_sel)

            rcA, reA = self._mem_read_cost(memA, a_bits_total, src_layer=layerA)
            rcB, reB = self._mem_read_cost(memB, b_bits_total, src_layer=layerB)
            wc, we  = self._mem_write_cost(memC, c_bits_total, src_layer=layerC)
            c_in = max(rcA, rcB)

            self._acc_rw(memA, c_in, reA, is_read=True)
            self._acc_rw(memB, 0, reB, is_read=True)
            self._acc_comp(cu_sel, cc, ec)
            self._acc_rw(memC, wc, we, is_read=False)

            cycles = max(c_in, cc, wc)
            energy = reA + reB + ec + we
            bits_read = a_bits_total + b_bits_total
            bits_written = c_bits_total

        elif ttype == 'ucieop':
            op = item['op']
            cycles = math.ceil(op.size_bits / max(1, self.ucie_bandwidth))  # 用 ceil 避免被截断为 0
            energy = op.size_bits * self.ucie_energy_per_bit * 1e-3         # pJ → nJ
            
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
