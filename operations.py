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

class MatMul(Op):
    def __init__(self, A: str, B: str, C: str):
        super().__init__(f"MatMul:{A}x{B}->{C}")
        self.A, self.B, self.C = A, B, C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        M, K = shapes[self.A].dims
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
        assert shapes[self.A].dims == shapes[self.B].dims
        return int(np.prod(shapes[self.A].dims))

    def required_tensors(self) -> List[str]:
        return [self.A, self.B]

class SoftmaxOp(Op):
    def __init__(self, input_tensor: str, axis: int, output_tensor: str):
        super().__init__(f"Softmax:{input_tensor}-> {output_tensor} axis={axis}")
        self.input, self.axis, self.output = input_tensor, axis, output_tensor

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        return int(np.prod(shapes[self.input].dims))

    def required_tensors(self) -> List[str]:
        return [self.input]

class LayerNorm(Op):
    def __init__(self, A: str, C: str):
        super().__init__(f"LayerNorm:{A}->{C}")
        self.A, self.C = A, C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        return int(np.prod(shapes[self.A].dims))

    def required_tensors(self) -> List[str]:
        return [self.A]

class Attention(Op):
    def __init__(self, Q: str, K: str, V: str, out: str):
        super().__init__(f"Attention:{Q},{K},{V}->{out}")
        self.Q, self.K, self.V, self.out = Q, K, V, out

    def required_tensors(self) -> List[str]:
        return [self.Q, self.K, self.V]