# FILE: operations.py
# -------------------
# Defines the neural network operations as classes.

import numpy as np
from typing import List, Dict
from datatypes import TensorShape

class Op:
    """Base class for all operations."""
    def __init__(self, name: str):
        self.name = name

    def required_tensors(self) -> List[str]:
        return []

class MatMul(Op):
    """Matrix Multiplication operation."""
    def __init__(self, A: str, B: str, C: str):
        super().__init__(f"MatMul:{A}x{B}->{C}")
        self.A = A
        self.B = B
        self.C = C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        a = shapes[self.A].dims
        b = shapes[self.B].dims
        M, K = a[-2], a[-1]
        K2, N = b[-2], b[-1]
        assert K == K2
        return M * N * K

    def required_tensors(self) -> List[str]:
        return [self.A, self.B]

class PatchEmbed(Op):
    """Patch Embedding operation for Vision Transformers."""
    def __init__(self, input_img: str, patch_w: int, patch_h: int, out_name: str, weight_name: str):
        super().__init__(f"PatchEmbed:{input_img}->{out_name}")
        self.input_img = input_img
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.out_name = out_name
        self.weight_name = weight_name

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        img_shape = shapes[self.input_img].dims
        C, H, W = img_shape
        n_patches_h = H // self.patch_h
        n_patches_w = W // self.patch_w
        num_patches = n_patches_h * n_patches_w
        patch_size = C * self.patch_h * self.patch_w
        out_dim = shapes[self.weight_name].dims[1]
        return num_patches * patch_size * out_dim

    def required_tensors(self) -> List[str]:
        return [self.input_img, self.weight_name]

class UnaryOp(Op):
    """Element-wise unary operations (e.g., GELU, EXP)."""
    def __init__(self, kind: str, A: str, C: str):
        super().__init__(f"{kind}:{A}->{C}")
        self.kind = kind.upper()
        self.A = A
        self.C = C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        return int(np.prod(shapes[self.A].dims))

    def required_tensors(self) -> List[str]:
        return [self.A]

class BinaryOp(Op):
    """Element-wise binary operations (e.g., ADD, DIV)."""
    def __init__(self, kind: str, A: str, B: str, C: str):
        super().__init__(f"{kind}:{A},{B}->{C}")
        self.kind = kind.upper()
        self.A = A
        self.B = B
        self.C = C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        assert shapes[self.A].dims == shapes[self.B].dims
        return int(np.prod(shapes[self.A].dims))

    def required_tensors(self) -> List[str]:
        return [self.A, self.B]

class SoftmaxOp(Op):
    """Softmax operation."""
    def __init__(self, input_tensor: str, axis: int, output_tensor: str):
        super().__init__(f"Softmax:{input_tensor}-> {output_tensor} axis={axis}")
        self.input = input_tensor
        self.axis = axis
        self.output = output_tensor

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        return int(np.prod(shapes[self.input].dims))

    def required_tensors(self) -> List[str]:
        return [self.input]

class LayerNorm(Op):
    """Layer Normalization operation."""
    def __init__(self, A: str, C: str):
        super().__init__(f"LayerNorm:{A}->{C}")
        self.A = A
        self.C = C

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        return int(np.prod(shapes[self.A].dims))

    def required_tensors(self) -> List[str]:
        return [self.A]

class Attention(Op):
    """High-level Attention operation (decomposed by the compiler)."""
    def __init__(self, Q: str, K: str, V: str, out: str):
        super().__init__(f"Attention:{Q},{K},{V}->{out}")
        self.Q = Q
        self.K = K
        self.V = V
        self.out = out

    def required_tensors(self) -> List[str]:
        return [self.Q, self.K, self.V]