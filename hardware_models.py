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

# ------------ ACU model (for complex elementwise & nonlinear ops) -------------

@dataclass
class ACU:
    throughput_elements_per_cycle: int
    energy_per_element_nj: float
    call_latency_cycles: int = 1
    op_cost_multiplier: Dict[str, float] = field(default_factory=lambda: {
        'NEG': 0.2, 'DIV': 2.0, 'EXP': 5.0, 'GELU': 4.0, 'SILU': 3.0,
        'SOFTMAX': 6.0, 'LAYERNORM': 6.0,
    })

    def op_cycles_and_energy(self, op_name: str, elements: int) -> Tuple[int, float]:
        m = self.op_cost_multiplier.get(op_name.upper(), 1.0)
        cycles = math.ceil(elements / self.throughput_elements_per_cycle) + self.call_latency_cycles
        energy = elements * self.energy_per_element_nj * m
        return cycles, energy

# ----------------------------- Simulator Stats -----------------------------

@dataclass
class Stats:
    cycles: int = 0
    energy_nj: float = 0.0
    macs: int = 0
    bits_read: int = 0
    bits_written: int = 0
    breakdown: Dict[str, float] = field(default_factory=dict)