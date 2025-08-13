# FILE: datatypes.py
# ------------------
# Contains basic data structures for tensors, hardware components, and statistics.

from dataclasses import dataclass, field
import math
import numpy as np
from typing import Tuple, Dict

# ----------------------------- Basic datatypes (bit-accurate) -----------------------------

@dataclass
class TensorShape:
    """Represents the dimensions of a tensor."""
    dims: Tuple[int, ...]

@dataclass
class Tensor:
    """Represents a tensor, including its shape, size, and device placement."""
    name: str
    shape: TensorShape
    size_bits: int
    device: str = "dram"  # Can be 'dram' or 'rram'

    @property
    def size_bytes(self) -> int:
        """Calculates the size of the tensor in bytes."""
        return (self.size_bits + 7) // 8

# ----------------------------- Memory models (bit units) -----------------------------

@dataclass
class MemoryDevice:
    """Models a memory device with its capacity, bandwidth, and energy costs."""
    name: str
    capacity_bits: int
    read_bw_bits_per_cycle: int
    write_bw_bits_per_cycle: int
    read_energy_per_bit: float
    write_energy_per_bit: float
    access_latency_cycles: int
    used_bits: int = 0

    def can_allocate(self, size_bits: int) -> bool:
        """Checks if a tensor of a given size can be allocated."""
        return self.used_bits + size_bits <= self.capacity_bits

    def allocate(self, size_bits: int) -> bool:
        """Allocates space on the device if available."""
        if self.can_allocate(size_bits):
            self.used_bits += size_bits
            return True
        return False

    def free(self, size_bits: int) -> None:
        """Frees up space on the device."""
        self.used_bits = max(0, self.used_bits - size_bits)

# ----------------------------- Compute unit model -----------------------------

@dataclass
class ComputeUnit:
    """Models a basic compute unit (e.g., for MAC operations)."""
    macs_per_cycle: int
    energy_per_mac_nj: float  # Energy per MAC in nanojoules

# ------------ ACU model (for complex elementwise & nonlinear ops) -------------

@dataclass
class ACU:
    """Models an Auxiliary Compute Unit for special functions (e.g., GELU, Softmax)."""
    throughput_elements_per_cycle: int
    energy_per_element_nj: float
    call_latency_cycles: int = 1
    op_cost_multiplier: Dict[str, float] = field(default_factory=lambda: {
        'NEG': 0.2, 'DIV': 2.0, 'EXP': 5.0, 'GELU': 4.0, 'SILU': 3.0,
        'SOFTMAX': 6.0, 'LAYERNORM': 6.0,
    })

    def op_cycles_and_energy(self, op_name: str, elements: int) -> Tuple[int, float]:
        """Calculates the cycles and energy for a given operation."""
        m = self.op_cost_multiplier.get(op_name.upper(), 1.0)
        cycles = math.ceil(elements / self.throughput_elements_per_cycle) + self.call_latency_cycles
        energy = elements * self.energy_per_element_nj * m
        return cycles, energy

# ----------------------------- Simulator Statistics -----------------------------

@dataclass
class Stats:
    """Holds the simulation results."""
    cycles: int = 0
    energy_nj: float = 0.0
    macs: int = 0
    bits_read: int = 0
    bits_written: int = 0
    breakdown: Dict[str, float] = field(default_factory=dict)