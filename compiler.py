# FILE: compiler.py
# -----------------
# Implements the compiler, which maps tensors to memory and schedules operations.

import math
from typing import List, Dict, Any
from datatypes import MemoryDevice, ComputeUnit
from model import Model
from operations import MatMul, PatchEmbed, LayerNorm, Attention, SoftmaxOp, UnaryOp, BinaryOp

class SimpleCompiler:
    """
    A simple compiler that places tensors and creates an execution schedule.
    It decomposes high-level ops like Attention into smaller, simulatable steps.
    """
    def __init__(self, model: Model, rram: MemoryDevice, dram: MemoryDevice, cu: ComputeUnit,
                 bits_per_element=32, tile_K=256, tile_M=64, tile_N=64):
        self.model = model
        self.rram = rram
        self.dram = dram
        self.cu = cu
        self.tile_K = tile_K
        self.tile_M = tile_M
        self.tile_N = tile_N
        self.bpe_bits = bits_per_element

    def place(self) -> Dict[str, str]:
        """Places tensors onto memory devices based on simple heuristics."""
        placements = {}
        for tname, tensor in self.model.tensors.items():
            # Prioritize placing weights on RRAM if possible
            if tname.startswith('W') or tensor.device == 'rram':
                if self.rram.allocate(tensor.size_bits):
                    placements[tname] = 'rram'
                    tensor.device = 'rram'
                else: # Fallback to DRAM
                    placements[tname] = 'dram'
                    tensor.device = 'dram'
            else: # Place activations on DRAM by default
                if self.dram.allocate(tensor.size_bits):
                    placements[tname] = 'dram'
                    tensor.device = 'dram'
                else: # Fallback to RRAM
                    placements[tname] = 'rram'
                    tensor.device = 'rram'
        return placements

    def compile(self) -> List[Dict[str, Any]]:
        """Generates an executable schedule from the model's operations."""
        placements = self.place()
        schedule = []
        for op in self.model.ops:
            if isinstance(op, MatMul):
                # Tile MatMul operations
                dims = self.model.shapes[op.A].dims
                M, K = dims[-2], dims[-1]
                K2, N = self.model.shapes[op.B].dims
                assert K == K2
                for m0 in range(0, M, self.tile_M):
                    msize = min(self.tile_M, M - m0)
                    for n0 in range(0, N, self.tile_N):
                        nsize = min(self.tile_N, N - n0)
                        for k0 in range(0, K, self.tile_K):
                            ksize = min(self.tile_K, K - k0)
                            schedule.append({
                                'op': op, 'type': 'matmul_tile',
                                'm0': m0, 'n0': n0, 'k0': k0,
                                'msize': msize, 'nsize': nsize, 'ksize': ksize
                            })
            # Decompose high-level ops into simpler types for the simulator
            elif isinstance(op, Attention):
                schedule.append({'op': op, 'type': 'attention'})
            # Pass-through for ops understood by the ACU/Simulator
            elif isinstance(op, PatchEmbed):
                schedule.append({'op': op, 'type': 'patch_embed'})
            elif isinstance(op, LayerNorm):
                schedule.append({'op': op, 'type': 'layernorm'})
            elif isinstance(op, SoftmaxOp):
                schedule.append({'op': op, 'type': 'softmax'})
            elif isinstance(op, UnaryOp):
                schedule.append({'op': op, 'type': 'unary'})
            elif isinstance(op, BinaryOp):
                schedule.append({'op': op, 'type': 'binary'})
            else:
                schedule.append({'op': op, 'type': 'unknown'})
        return schedule