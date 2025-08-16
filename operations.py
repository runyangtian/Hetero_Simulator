# 定义了所有神经网络的算子（操作），如矩阵乘法、注意力等。

import numpy as np
from typing import List, Dict
from hardware_models import TensorShape

# ----------------------------- Operation models -----------------------------

class Op:
    def __init__(self, name: str):
        self.name = name

    def required_tensors(self) -> List[str]:
        return []

class UnaryOp(Op):
    def __init__(self, kind: str, A: str, C: str):
        super().__init__(f"{kind}:{A}->{C}")
        self.kind, self.A, self.C = kind.upper(), A, C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        return int(np.prod(shapes[self.A].dims))

    def required_tensors(self) -> List[str]:
        return [self.A]

class BinaryOp(Op):
    def __init__(self, kind: str, A: str, B: str, C: str):
        super().__init__(f"{kind}:{A},{B}->{C}")
        self.kind, self.A, self.B, self.C = kind.upper(), A, B, C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        dimsA = shapes[self.A].dims
        dimsB = shapes[self.B].dims     
        # assert shapes[self.A].dims == shapes[self.B].dims
        assert dimsA[-2:] == dimsB[-2:], f"BinaryOp input shapes not compatible: {dimsA} vs {dimsB}"
        return int(np.prod(shapes[self.A].dims))

    def required_tensors(self) -> List[str]:
        return [self.A, self.B]

class MatMul(Op):
    def __init__(self, A: str, B: str, C: str, transpose_B: bool = False):
        super().__init__(f"MatMul:{A}x{B}->{C}")
        self.A, self.B, self.C = A, B, C
        self.transpose_B = transpose_B

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        M, K = shapes[self.A].dims
        if self.transpose_B:
            N, K2 = shapes[self.B].dims   # 转置情况下 B 是 (N, K)
        else:
            K2, N = shapes[self.B].dims
        assert K == K2
        return M * N * K

    def required_tensors(self) -> List[str]:
        return [self.A, self.B]


class PatchEmbed(Op):
    def __init__(self, input_img: str, patch_w: int, patch_h: int, out_name: str, weight_name: str):
        super().__init__(f"PatchEmbed:{input_img}->{out_name}")
        self.input_img, self.patch_w, self.patch_h = input_img, patch_w, patch_h
        self.out_name, self.weight_name = out_name, weight_name

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        C, H, W = shapes[self.input_img].dims
        n_patches_h, n_patches_w = H // self.patch_h, W // self.patch_w
        num_patches = n_patches_h * n_patches_w
        patch_size = C * self.patch_h * self.patch_w
        out_dim = shapes[self.weight_name].dims[1]
        return num_patches * patch_size * out_dim

    def required_tensors(self) -> List[str]:
        return [self.input_img, self.weight_name]

class SoftmaxOp(UnaryOp):
    def __init__(self, input_tensor: str, axis: int, output_tensor: str):
        super().__init__('SOFTMAX', input_tensor, output_tensor)
        self.axis = axis

class LayerNorm(UnaryOp):
    def __init__(self, A: str, C: str):
        super().__init__('LAYERNORM', A, C)

class GeluOp(UnaryOp):
    def __init__(self, A: str, C: str):
        super().__init__('GELU', A, C)

class ReluOp(UnaryOp):
    def __init__(self, A: str, C: str):
        super().__init__('RELU', A, C)

class SigmoidOp(UnaryOp):
    def __init__(self, A: str, C: str):
        super().__init__('SIGMOID', A, C)

class TanhOp(UnaryOp):
    def __init__(self, A: str, C: str):
        super().__init__('TANH', A, C)

class AddOp(BinaryOp):
    def __init__(self, A: str, B: str, C: str):
        super().__init__('ADD', A, B, C)

class SubOp(BinaryOp):
    def __init__(self, A: str, B: str, C: str):
        super().__init__('SUB', A, B, C)

class MulOp(BinaryOp):
    def __init__(self, A: str, B: str, C: str):
        super().__init__('MUL', A, B, C)

class DivOp(BinaryOp):
    def __init__(self, A: str, B: str, C: str):
        super().__init__('DIV', A, B, C)

class UCIeOp:
    def __init__(self, size_bits):
        self.size_bits = size_bits

    def __repr__(self):
        return f"UCIeOp(bits={self.size_bits})"

class ParallelOp(Op):
    def __init__(self, branches: List[Op]):
        super().__init__(f"Parallel({','.join(b.name for b in branches)})")
        self.branches = branches

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        # 这里是逻辑 FLOPs 总和（Q,K,V 的 FLOPs 累加）
        return sum(b.flops(shapes) for b in self.branches)

    def required_tensors(self) -> List[str]:
        # 所有分支需要的张量集合
        reqs = []
        for b in self.branches:
            reqs.extend(b.required_tensors())
        return reqs
