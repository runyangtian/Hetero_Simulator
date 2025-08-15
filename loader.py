# 负责解析 JSON 文件并构建 Model 对象。

import json
from typing import Dict, Any

from model import Model
from operations import (
    MatMul, PatchEmbed, LayerNorm,
    SoftmaxOp, GeluOp, ReluOp, SigmoidOp, TanhOp,
    AddOp, SubOp, MulOp, DivOp, UCIeOp
)


class JSONModelLoader:
    def __init__(self, default_bits: int = 16):
        self.default_bits = default_bits

    def build(self, spec: Dict[str, Any]) -> Model:
        m = Model()
        # 添加 tensor
        for t in spec.get("tensors", []):
            m.add_tensor(
                name=t["name"],
                shape=tuple(int(x) for x in t["shape"]),
                bits_per_element=int(t.get("bits", self.default_bits)),
                device=t.get("device", "dram")
            )

        # 添加算子
        for o in spec.get("ops", []):
            tpe = o.get("type", "").lower()
            if tpe == 'matmul':
                m.add_op(MatMul(o['A'], o['B'], o['C']))
            elif tpe == 'patchembed':
                m.add_op(PatchEmbed(o['input_img'], int(o['patch_w']), int(o['patch_h']), o['out'], o['weight_name']))
            elif tpe == 'layernorm':
                m.add_op(LayerNorm(o['A'], o['C']))
            elif tpe == 'geluop':
                m.add_op(GeluOp(o['A'], o['C']))
            elif tpe == 'addop':
                m.add_op(AddOp(o['A'], o['B'], o['C']))
            elif tpe == 'softmaxop':
                m.add_op(SoftmaxOp(o['input'], int(o.get('axis', -1)), o['output']))
            elif tpe == 'ucieop':
                m.add_op(UCIeOp(o['src'], o['dst'], int(o['size_bits'])))
            else:
                raise ValueError(f"Unknown op type in JSON: {o}")
        return m
