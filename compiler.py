# file: compiler.py
# 将高级模型描述转换为底层的、可模拟的指令序列。

from typing import List, Dict, Any
from model import Model, Op
from hardware_models import MemoryDevice, ComputeUnit
from operations import (
    MatMul, Conv2D, AvgPool2D, LayerNorm,
    SoftmaxOp, GeluOp, ReluOp, SigmoidOp, TanhOp,
    AddOp, SubOp, MulOp, DivOp, UCIeOp, ParallelOp
)

class SimpleCompiler:
    def __init__(self, model: Model, rram: MemoryDevice, dram: MemoryDevice, tile_K=256, tile_M=128, tile_N=128):
        self.model = model
        self.rram, self.dram = rram, dram
        self.tile_K, self.tile_M, self.tile_N = tile_K, tile_M, tile_N

    def place(self) -> Dict[str, str]:
        placements = {}
        for tname, tensor in self.model.tensors.items():
            if tensor.device == 'rram':
                if self.rram.allocate(tensor.size_bits):
                    placements[tname] = 'rram'
                    tensor.device = 'rram'
                else: # Fallback to DRAM if RRAM is full
                    print("RRAM is full")
                    placements[tname] = 'dram'
                    tensor.device = 'dram'
            else:
                if self.dram.allocate(tensor.size_bits):
                    placements[tname] = 'dram'
                    tensor.device = 'dram'
                else: # Fallback to RRAM
                    print("DRAM is full")
                    placements[tname] = 'rram'
                    tensor.device = 'rram'
        return placements

    def _compile_op(self, op: Op) -> List[Dict[str, Any]]:
        """编译单个算子，返回一个 schedule item 列表"""
        schedule = []
        # MatMul 需要 tiling
        if isinstance(op, MatMul):
            M, K = self.model.shapes[op.A].dims
            if op.transpose_B:
                N, K2 = self.model.shapes[op.B].dims
            else:
                K2, N = self.model.shapes[op.B].dims
            # print(M, K, N, K2)
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
        # ParallelOp 需要递归编译其分支
        elif isinstance(op, ParallelOp):
            compiled_branches = []
            for branch_op in op.branches:
                # 递归调用来编译每个分支
                branch_schedule = self._compile_op(branch_op)
                compiled_branches.append(branch_schedule)
            
            schedule.append({
                'op': op,
                'type': 'parallel',
                'branches': compiled_branches
            })

        # 其他具体算子直接用类名作为 type
        else:
            schedule.append({
                'op': op,
                'type': op.__class__.__name__.lower()
            })
        
        return schedule

    def compile(self) -> List[Dict[str, Any]]:
        self.place()
        final_schedule = []
        for op in self.model.ops:
            final_schedule.extend(self._compile_op(op))
        return final_schedule