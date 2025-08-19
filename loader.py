# 负责解析 JSON 文件并构建 Model 对象。

import json
from typing import Dict, Any

from model import Model
from operations import (
    MatMul, PatchEmbed, LayerNorm,
    SoftmaxOp, GeluOp, ReluOp, SigmoidOp, TanhOp,
    AddOp, SubOp, MulOp, DivOp, UCIeOp, ParallelOp
)

def _assign_layer(t: Dict[str, Any]) -> int:
    n = t["name"]
    for pat, ly in [
        (r'^(W_q|W_k|W_v|W1|W2)_', 1),
        (r'^(tokens|Q_0|K_0|V_0)$', 2),
        (r'^attn_out_', 3),
        (r'^(norm1_out|resid1|ff2)_', 4),
        (r'^(norm2_out|resid2|ff1(_act)?)_', 5),
        (r'^attn_(scores|probs)_', 6),
        (r'^W_embed$', 10),
        (r'^img$', 12),
        (r'^Q_remote$', 25),
    ]:
        if re.search(pat, n):
            return ly
    return 15  # 默认中间层

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
                device=t.get("device")
                # layer=t.get("layer", _assign_layer(t))
            )

        # 添加算子
        for o in spec.get("ops", []):
            tpe = o.get("type", "").lower()
            if tpe == 'matmul':
                transpose_B = o.get('transpose_B', False)   # 默认 False
                m.add_op(MatMul(o['A'], o['B'], o['C'], transpose_B=transpose_B))
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
                m.add_op(UCIeOp(int(o['size_bits'])))
            elif tpe == 'parallelops':
                branches = []
                for branch in o['branches']:
                    # 递归调用自己，把 branch 解析成 Op
                    branches.append(self.build({"ops": [branch], "tensors": []}).ops[0])
                m.add_op(ParallelOp(branches))
            else:
                raise ValueError(f"Unknown op type in JSON: {o}")
        return m
