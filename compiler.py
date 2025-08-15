# 将高级模型描述转换为底层的、可模拟的指令序列。

from typing import List, Dict, Any
from model import Model
from hardware_models import MemoryDevice, ComputeUnit
from operations import (
    MatMul, PatchEmbed, LayerNorm,
    SoftmaxOp, GeluOp, ReluOp, SigmoidOp, TanhOp,
    AddOp, SubOp, MulOp, DivOp, UCIeOp
)

class SimpleCompiler:
    def __init__(self, model: Model, rram: MemoryDevice, dram: MemoryDevice, cu: ComputeUnit,
                 bits_per_element=32, tile_K=256, tile_M=64, tile_N=64):
        self.model = model
        self.rram, self.dram, self.cu = rram, dram, cu
        self.tile_K, self.tile_M, self.tile_N = tile_K, tile_M, tile_N
        self.bpe_bits = bits_per_element

    def place(self) -> Dict[str, str]:
        placements = {}
        for tname, tensor in self.model.tensors.items():
            if tname.startswith('W') or tensor.device == 'rram':
                if self.rram.allocate(tensor.size_bits):
                    placements[tname] = 'rram'
                    tensor.device = 'rram'
                else: # Fallback to DRAM if RRAM is full
                    placements[tname] = 'dram'
                    tensor.device = 'dram'
            else:
                if self.dram.allocate(tensor.size_bits):
                    placements[tname] = 'dram'
                    tensor.device = 'dram'
                else: # Fallback to RRAM
                    placements[tname] = 'rram'
                    tensor.device = 'rram'
        return placements

    def compile(self) -> List[Dict[str, Any]]:
        self.place()
        schedule = []

        for op in self.model.ops:
            # MatMul 仍然需要 tiling
            if isinstance(op, MatMul):
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
                                'op': op,
                                'type': 'matmul_tile',
                                'm0': m0, 'n0': n0, 'k0': k0,
                                'msize': msize, 'nsize': nsize, 'ksize': ksize,
                                'A_dev': self.model.tensors[op.A].device,
                                'B_dev': self.model.tensors[op.B].device
                            })
            # 所有具体算子直接用类名作为 type
            else:
                schedule.append({
                    'op': op,
                    'type': op.__class__.__name__.lower()
                })

        return schedule
