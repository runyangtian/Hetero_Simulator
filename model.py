# Container for all tensors and ops

from typing import Dict, List, Tuple
import numpy as np
from hardware_models import Tensor, TensorShape
from operations import Op

class Model:
    def __init__(self):
        self.tensors: Dict[str, Tensor] = {}
        self.shapes: Dict[str, TensorShape] = {}
        self.ops: List[Op] = []

    def add_tensor(self, name: str, shape: Tuple[int, ...], bits_per_element=32, device='dram', layer: int=-1):
        shape_obj = TensorShape(shape)
        size_bits = int(np.prod(shape) * bits_per_element)
        self.tensors[name] = Tensor(name, shape_obj, size_bits, device, layer, bits_per_element)
        self.shapes[name] = shape_obj

    def add_op(self, op: Op):
        self.ops.append(op)