# 负责解析 JSON 文件并构建 Model 对象。

import json
from typing import Dict, Any

from model import Model
from operations import (
    MatMul, PatchEmbed, LayerNorm, Attention, UnaryOp, BinaryOp, SoftmaxOp
)

class JSONModelLoader:
    """Builds a Model from a JSON description."""
    def __init__(self, default_bits: int = 16):
        self.default_bits = default_bits

    def build(self, spec: Dict[str, Any]) -> Model:
        m = Model()
        for t in spec.get("tensors", []):
            m.add_tensor(
                name=t["name"],
                shape=tuple(int(x) for x in t["shape"]),
                bits_per_element=int(t.get("bits", self.default_bits)),
                device=t.get("device", "dram")
            )

        for o in spec.get("ops", []):
            tpe = o.get("type", "").lower()
            if tpe == 'matmul':
                m.add_op(MatMul(o['A'], o['B'], o['C']))
            elif tpe == 'patchembed':
                m.add_op(PatchEmbed(o['input_img'], int(o['patch_w']), int(o['patch_h']), o['out'], o['weight']))
            elif tpe == 'layernorm':
                m.add_op(LayerNorm(o['A'], o['C']))
            elif tpe == 'attention':
                m.add_op(Attention(o['Q'], o['K'], o['V'], o['out']))
            elif tpe == 'unary':
                m.add_op(UnaryOp(o['kind'], o['A'], o['C']))
            elif tpe == 'binary':
                m.add_op(BinaryOp(o['kind'], o['A'], o['B'], o['C']))
            elif tpe == 'softmax':
                m.add_op(SoftmaxOp(o['input'], int(o.get('axis', -1)), o['output']))
            else:
                raise ValueError(f"Unknown op type in JSON: {o}")
        return m