import math
import numpy as np
from typing import List, Dict, Any
from datatypes import Stats, MemoryDevice, ComputeUnit, ACU
from model import Model
from operations import Op, PatchEmbed, LayerNorm, Attention, SoftmaxOp, UnaryOp, BinaryOp

class Simulator:
    """
    Executes a compiled schedule to simulate hardware performance, tracking
    cycles, energy, and data movement.
    """
    def __init__(self, model: Model, schedule: List[Dict[str, Any]], rram: MemoryDevice, dram: MemoryDevice, cu: ComputeUnit, acu: ACU, bits_per_element=32):
        self.model = model
        self.schedule = schedule
        self.rram = rram
        self.dram = dram
        self.cu = cu
        self.acu = acu
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

    def run(self) -> Stats:
        """Runs the simulation for all items in the schedule."""
        for item in self.schedule:
            ttype = item['type']
            op = item.get('op')
            
            if ttype == 'matmul_tile':
                m, n, k = item['msize'], item['nsize'], item['ksize']
                A_tile_bits = m * k * self.bpe_bits
                B_tile_bits = k * n * self.bpe_bits
                memA = self.rram if self.model.tensors[op.A].device == 'rram' else self.dram
                memB = self.rram if self.model.tensors[op.B].device == 'rram' else self.dram
                
                cA, eA = self._mem_read_cost(memA, A_tile_bits)
                cB, eB = self._mem_read_cost(memB, B_tile_bits)
                
                macs = m * n * k
                cc, ec = self._compute_cost(macs)
                
                C_tile_bits = m * n * self.bpe_bits
                cW, eW = self._mem_write_cost(self.dram, C_tile_bits) # Assume outputs go to DRAM
                
                cycles = max(cA + cB, cc) + cW
                energy = eA + eB + ec + eW
                
                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.macs += macs
                self.stats.bits_read += (A_tile_bits + B_tile_bits)
                self.stats.bits_written += C_tile_bits
                self.stats.breakdown['matmul'] = self.stats.breakdown.get('matmul', 0) + energy

            elif ttype == 'attention':
                op: Attention = op
                L, d = self.model.shapes[op.Q].dims[-2:]
                
                # QK^T
                macs_qk = L * L * d
                c_qk, e_qk = self._compute_cost(macs_qk)
                q_bits, k_bits, v_bits = [self.model.tensors[t].size_bits for t in (op.Q, op.K, op.V)]
                rq_c, rq_e = self._mem_read_cost(self.dram, q_bits)
                rk_c, rk_e = self._mem_read_cost(self.dram, k_bits)
                scores_bits = L * L * self.bpe_bits
                wsc_c, wsc_e = self._mem_write_cost(self.dram, scores_bits)
                
                # Softmax
                soft_cycles, soft_energy = self.acu.op_cycles_and_energy('SOFTMAX', L * L)
                
                # Scores * V
                macs_sv = L * L * d
                c_sv, e_sv = self._compute_cost(macs_sv)
                rv_c, rv_e = self._mem_read_cost(self.dram, v_bits)
                out_bits = L * d * self.bpe_bits
                wo_c, wo_e = self._mem_write_cost(self.dram, out_bits)
                
                # Combine costs
                cycles = max(rq_c + rk_c, c_qk) + wsc_c + soft_cycles + max(rv_c, c_sv) + wo_c
                energy = rq_e + rk_e + e_qk + wsc_e + soft_energy + rv_e + e_sv + wo_e
                
                self.stats.cycles += cycles
                self.stats.energy_nj += energy
                self.stats.macs += (macs_qk + macs_sv)
                self.stats.bits_read += (q_bits + k_bits + v_bits + scores_bits)
                self.stats.bits_written += (scores_bits + out_bits)
                self.stats.breakdown['attention'] = self.stats.breakdown.get('attention', 0) + energy

            elif ttype in ['layernorm', 'softmax'] or isinstance(op, (UnaryOp, BinaryOp)):
                op_kind = ttype.upper() if ttype in ['layernorm', 'softmax'] else op.kind
                
                # Determine inputs and outputs
                if isinstance(op, BinaryOp):
                    input_tensors = [op.A, op.B]
                else:
                    input_tensors = [op.A if hasattr(op, 'A') else op.input]
                
                output_tensor_name = op.C if hasattr(op, 'C') else op.output
                
                # Calculate costs
                elems = int(np.prod(self.model.shapes[input_tensors[0]].dims))
                acu_c, acu_e = self.acu.op_cycles_and_energy(op_kind, elems)
                
                total_read_c, total_read_e = 0, 0
                total_read_bits = 0
                for t_name in input_tensors:
                    t_bits = self.model.tensors[t_name].size_bits
                    dev = self.rram if self.model.tensors[t_name].device == 'rram' else self.dram
                    read_c, read_e = self._mem_read_cost(dev, t_bits)
                    total_read_c += read_c
                    total_read_e += read_e
                    total_read_bits += t_bits
                    
                out_bits = self.model.tensors[output_tensor_name].size_bits
                write_c, write_e = self._mem_write_cost(self.dram, out_bits)
                
                total_cycles = max(total_read_c, acu_c) + write_c
                total_energy = total_read_e + write_e + acu_e
                
                self.stats.cycles += total_cycles
                self.stats.energy_nj += total_energy
                self.stats.bits_read += total_read_bits
                self.stats.bits_written += out_bits
                self.stats.breakdown[op_kind.lower()] = self.stats.breakdown.get(op_kind.lower(), 0) + total_energy

            else:
                self.stats.cycles += 1 # Default cost for unknown ops
                
        return self.stats