# Define all ops

import numpy as np
from typing import List, Dict
from hardware_models import TensorShape

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

class Conv2D(Op):
    def __init__(self, input_img: str, weight_name: str, out_name: str,
                 kernel_h: int, kernel_w: int,
                 stride_h: int = 1, stride_w: int = 1,
                 padding_h: int = 0, padding_w: int = 0,
                 groups: int = 1):
        super().__init__(f"Conv2D:{input_img}->{out_name}")
        self.input_img = input_img
        self.weight_name = weight_name
        self.out_name = out_name
        self.kh, self.kw = kernel_h, kernel_w
        self.sh, self.sw = stride_h, stride_w
        self.ph, self.pw = padding_h, padding_w
        self.groups = groups

    def _out_hw(self, H: int, W: int) -> (int, int):
        Ho = (H + 2*self.ph - self.kh) // self.sh + 1
        Wo = (W + 2*self.pw - self.kw) // self.sw + 1
        return Ho, Wo

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        C_in, H, W = shapes[self.input_img].dims
        C_out = shapes[self.weight_name].dims[0]
        Cin_per_g = C_in // max(1, self.groups)
        Ho, Wo = self._out_hw(H, W)
        macs = C_out * Ho * Wo * Cin_per_g * self.kh * self.kw
        return macs

    def required_tensors(self) -> List[str]:
        return [self.input_img, self.weight_name]

class AvgPool2D(Op):
    def __init__(self, input_img: str, out_name: str,
                 kernel_h: int, kernel_w: int,
                 stride_h: int = 1, stride_w: int = 1,
                 padding_h: int = 0, padding_w: int = 0):
        super().__init__(f"AvgPool2D:{input_img}->{out_name}")
        self.input_img = input_img
        self.out_name = out_name
        self.kh, self.kw = kernel_h, kernel_w
        self.sh, self.sw = stride_h, stride_w
        self.ph, self.pw = padding_h, padding_w

    def _out_hw(self, H: int, W: int) -> (int, int):
        Ho = (H + 2*self.ph - self.kh) // self.sh + 1
        Wo = (W + 2*self.pw - self.kw) // self.sw + 1
        return Ho, Wo

    def flops(self, shapes: Dict[str, TensorShape]) -> int:
        C, H, W = shapes[self.input_img].dims
        Ho, Wo = self._out_hw(H, W)
        return C * Ho * Wo * self.kh * self.kw

    def required_tensors(self) -> List[str]:
        return [self.input_img]

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
        return sum(b.flops(shapes) for b in self.branches)

    def required_tensors(self) -> List[str]:
        reqs = []
        for b in self.branches:
            reqs.extend(b.required_tensors())
        return reqs
