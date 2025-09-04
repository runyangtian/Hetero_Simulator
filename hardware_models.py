# 包含所有硬件组件和基础数据结构的定义，比如张量、内存设备、计算单元和统计数据。

from dataclasses import dataclass, field
import math
import numpy as np
from typing import Tuple, Dict, List

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
    layer: int = -1
    bits_per_element: int = 16 

    @property
    def size_bytes(self) -> int:
        return (self.size_bits + 7) // 8

# ----------------------------- Memory models (bit units) -----------------------------

@dataclass
class MemoryDevice:
    name: str
    capacity_bits: int
    read_bw_bits_per_cycle: int   # bits per cycle read bandwidth
    write_bw_bits_per_cycle: int
    read_energy_per_bit: float    # nJ per bit read
    write_energy_per_bit: float   # nJ per bit write
    read_latency_cycles: int
    write_latency_cycles: int
    used_bits: int = 0
    # ====== 3D 分層與 TSV 建模 ======
    num_layers: int = 5
    logic_layer: int = 0

    # 帶寬（每個 cycle 可傳送幾多 bit）
    tsv_bw_bits_per_cycle: int = 262144          # 例：~128 GB/s@1GHz，可按需調
    # 固定底延遲：不論跨幾多層都有（介面/握手/同步等）
    tsv_base_latency_cycles: int = 3
    # 每 hop 的附加延遲（與 hop 數線性關係）
    tsv_fixed_latency_per_hop: int = 1

    def tsv_hops(self, src_layer: int, dst_layer: int = None) -> int:
        """計算 hop 數。如果 src_layer==0（邏輯層），則無需 TSV 傳輸"""
        if src_layer is None or src_layer < 0:
            return 0
        if src_layer == self.logic_layer:  # 在邏輯層，不經 TSV
            return 0
        dst = self.logic_layer if dst_layer is None else dst_layer
        return abs(int(src_layer) - int(dst))

    def tsv_cycles_for(self, size_bits: int, hops: int) -> int:
        """固定底延遲 + 串行化 + hop 延遲。如果 src_layer=0，就不加 TSV 延遲"""
        if size_bits <= 0 or hops == 0:  # 在邏輯層，無 TSV 延遲
            return 0
        ser = (size_bits + self.tsv_bw_bits_per_cycle - 1) // self.tsv_bw_bits_per_cycle
        return self.tsv_base_latency_cycles + ser + hops * self.tsv_fixed_latency_per_hop

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

# @dataclass
# class ComputeUnit: # aka PE
#     macs_per_cycle: int
#     energy_per_mac_nj: float      # energy per MAC in nJ

@dataclass
class ComputeUnit:
    name: str

    # MAC 引擎（matmul/conv/elementwise 等用 MAC 路徑時的吞吐/能耗）
    macs_per_cycle: int
    energy_per_mac_nj: float

    # SFE 引擎（softmax/gelu/layernorm…）的吞吐/能耗（以「元素/操作」為粒度）
    sfe_ops_per_cycle: int = 0
    sfe_energy_per_op_nj: float = 0.0

    # 支援的算子類別（僅標註用途，當前邏輯在 simulator 中選路）
    # supported_ops: List[str] = field(default_factory=list)

# ----------------------------- Simulator Stats -----------------------------

@dataclass
class Stats:
    cycles: int = 0
    energy_nj: float = 0.0
    macs: int = 0
    bits_read: int = 0
    bits_written: int = 0
    breakdown: Dict[str, float] = field(default_factory=dict)
    macs_breakdown: Dict[str, int] = field(default_factory=dict) 
    cycles_breakdown: Dict[str, int] = field(default_factory=dict) 