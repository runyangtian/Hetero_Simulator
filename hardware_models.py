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
    energy_per_mac_nj: float      # energy per MAC in nJ

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