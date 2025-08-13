import numpy as np
from typing import List, Tuple, Dict
from datatypes import Tensor, TensorShape
from operations import Op

class Model:
    """Represents a neural network as a collection of tensors and operations."""
    def __init__(self):
        self.tensors: Dict[str, Tensor] = {}
        self.shapes: Dict[str, TensorShape] = {}
        self.ops: List[Op] = []

    def add_tensor(self, name: str, shape: Tuple[int, ...], bits_per_element=32, device='dram'):
        """Adds a tensor to the model graph."""
        shape_obj = TensorShape(shape)
        size_bits = int(np.prod(shape) * bits_per_element)
        self.tensors[name] = Tensor(name, shape_obj, size_bits, device)
        self.shapes[name] = shape_obj

    def add_op(self, op: Op):
        """Adds an operation to the model graph."""
        self.ops.append(op)