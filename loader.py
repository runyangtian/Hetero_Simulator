# 负责解析 JSON 文件并构建 Model 对象。

import json
from typing import Dict, Any

from model import Model
from operations import (
    MatMul, Conv2D, AvgPool2D, LayerNorm,
    SoftmaxOp, GeluOp, ReluOp, SigmoidOp, TanhOp,
    AddOp, SubOp, MulOp, DivOp, UCIeOp, ParallelOp
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
                device=t.get("device"),
                layer=int(t.get("layer", -1))   # 新增，可默认 -1 表示未指定
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
            elif tpe == 'conv2d' or tpe == 'conv':   # 兼容 "Conv2D" / "Conv"
                # 允许 stride/padding 提供单值或 [h,w]
                def _pair(v):
                    if isinstance(v, list) or isinstance(v, tuple):
                        return int(v[0]), int(v[1])
                    else:
                        return int(v), int(v)
                kh, kw = int(o.get('kernel_h')), int(o.get('kernel_w'))
                sh, sw = _pair(o.get('stride', 1))
                ph, pw = _pair(o.get('padding', 0))
                groups = int(o.get('groups', 1))
                m.add_op(Conv2D(
                    input_img=o['input_img'],
                    weight_name=o['weight_name'],
                    out_name=o['out'],
                    kernel_h=kh, kernel_w=kw,
                    stride_h=sh, stride_w=sw,
                    padding_h=ph, padding_w=pw,
                    groups=groups
                ))
            elif tpe == 'avgpool2d':
                def _pair(v):
                    if isinstance(v, list) or isinstance(v, tuple):
                        return int(v[0]), int(v[1])
                    else:
                        return int(v), int(v)
                kh, kw = int(o.get('kernel_h')), int(o.get('kernel_w'))
                sh, sw = _pair(o.get('stride', 1))
                ph, pw = _pair(o.get('padding', 0))

                m.add_op(AvgPool2D(
                    input_img=o['input_img'],
                    out_name=o['out'],
                    kernel_h=kh, kernel_w=kw,
                    stride_h=sh, stride_w=sw,
                    padding_h=ph, padding_w=pw
                ))
            else:
                raise ValueError(f"Unknown op type in JSON: {o}")
        return m
