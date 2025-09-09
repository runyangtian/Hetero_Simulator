# analyze_pipeline.py
# 依次跑 6 个 *_A.json，按层数放大 encoder/decoder 两组，输出环节&算子两层统计 + 总量

import os
import json
import argparse
from collections import defaultdict

from hardware_models import MemoryDevice, ComputeUnit
from model import Model
from loader import JSONModelLoader
from compiler import SimpleCompiler
from simulator import Simulator

# stage_output_path = "./result/stages_summary_mobilevlm.csv"
# ops_output_path = "./result/ops_summary_mobilevlm.csv"
# total_output_path = "./result/totals_mobilevlm.txt"

stage_output_path = "./result/stages_summary_fastvlm.csv"
ops_output_path = "./result/ops_summary_fastvlm.csv"
total_output_path = "./result/totals_fastvlm.txt"

# —— 设备 & CU：保持与 main.py 一致 ——
def make_devices_and_cus():
    dram = MemoryDevice(
        name='3D_DRAM',
        capacity_bits = int(6.25*1024*1024*1024*8),    # 6.25GB
        read_bw_bits_per_cycle  = 1024,
        write_bw_bits_per_cycle = 1024,
        read_energy_per_bit  = 0.000429,
        write_energy_per_bit = 0.000429,
        read_latency_cycles = 3,
        write_latency_cycles = 3
    )

    rram = MemoryDevice(
        name='3D_RRAM',
        capacity_bits = int(2*1024*1024*1024*8),        # 2GB
        read_bw_bits_per_cycle  = 4096,
        write_bw_bits_per_cycle = 4096,
        read_energy_per_bit  = 0.0004,          # nJ/bit
        write_energy_per_bit = 0.00133,         # nJ/bit
        read_latency_cycles = 3,
        write_latency_cycles = 11
    )

    dram_cu = ComputeUnit(
        name='DRAM_CU',
        macs_per_cycle=32*32,        
        energy_per_mac_nj=0.000604,
        sfe_ops_per_cycle=256,
        sfe_energy_per_op_nj=0.00005,
    )

    rram_cu = ComputeUnit(
        name='RRAM_CU',
        macs_per_cycle=64*64,         
        energy_per_mac_nj=0.000604,
        sfe_ops_per_cycle=256,      
        sfe_energy_per_op_nj=0.00005,
    )

    return dram, rram, dram_cu, rram_cu

def run_one(json_path, freq_ghz=1.0, default_bits=16):
    dram, rram, dram_cu, rram_cu = make_devices_and_cus()
    with open(json_path, 'r') as f:
        spec = json.load(f)

    loader = JSONModelLoader(default_bits=default_bits)
    model: Model = loader.build(spec)

    compiler = SimpleCompiler(model, rram, dram, tile_K=128, tile_M=128, tile_N=128)
    schedule = compiler.compile()

    sim = Simulator(model, schedule, rram, dram, dram_cu, rram_cu)
    # sim = Simulator(model, schedule, dram, dram, dram_cu, dram_cu)

    stats = sim.run()

    return {
        "cycles": stats.cycles,
        "macs": stats.macs,
        "energy_nj": stats.energy_nj,
        "breakdown_energy_nj": dict(stats.breakdown),
        "breakdown_macs": dict(stats.macs_breakdown),
        "breakdown_cycles": dict(stats.cycles_breakdown),

        "cycles_read_dram":  stats.cycles_read_dram,
        "cycles_comp_dram":  stats.cycles_comp_dram,
        "cycles_write_dram": stats.cycles_write_dram,
        "cycles_read_rram":  stats.cycles_read_rram,
        "cycles_comp_rram":  stats.cycles_comp_rram,
        "cycles_write_rram": stats.cycles_write_rram,

        "energy_read_dram_nj":  stats.energy_read_dram_nj,
        "energy_comp_dram_nj":  stats.energy_comp_dram_nj,
        "energy_write_dram_nj": stats.energy_write_dram_nj,
        "energy_read_rram_nj":  stats.energy_read_rram_nj,
        "energy_comp_rram_nj":  stats.energy_comp_rram_nj,
        "energy_write_rram_nj": stats.energy_write_rram_nj,
    }

def scale_stats(s, k):
    out = {
        "cycles": s["cycles"] * k,
        "macs": s["macs"] * k,
        "energy_nj": s["energy_nj"] * k,
        "breakdown_energy_nj": {op: v * k for op, v in s["breakdown_energy_nj"].items()},
        "breakdown_macs": {op: v * k for op, v in s["breakdown_macs"].items()},
        "breakdown_cycles": {op: v * k for op, v in s["breakdown_cycles"].items()},

        "cycles_read_dram":  s["cycles_read_dram"]  * k,
        "cycles_comp_dram":  s["cycles_comp_dram"]  * k,
        "cycles_write_dram": s["cycles_write_dram"] * k,
        "cycles_read_rram":  s["cycles_read_rram"]  * k,
        "cycles_comp_rram":  s["cycles_comp_rram"]  * k,
        "cycles_write_rram": s["cycles_write_rram"] * k,

        "energy_read_dram_nj":  s["energy_read_dram_nj"]  * k,
        "energy_comp_dram_nj":  s["energy_comp_dram_nj"]  * k,
        "energy_write_dram_nj": s["energy_write_dram_nj"] * k,
        "energy_read_rram_nj":  s["energy_read_rram_nj"]  * k,
        "energy_comp_rram_nj":  s["energy_comp_rram_nj"]  * k,
        "energy_write_rram_nj": s["energy_write_rram_nj"] * k,
    }
    return out

def add_into(acc, s):
    acc["cycles"] += s["cycles"]
    acc["macs"] += s["macs"]
    acc["energy_nj"] += s["energy_nj"]
    for k, v in s["breakdown_energy_nj"].items():
        acc["breakdown_energy_nj"][k] += v
    for k, v in s["breakdown_macs"].items():
        acc["breakdown_macs"][k] += v
    for k, v in s["breakdown_cycles"].items():
        acc["breakdown_cycles"][k] += v

    acc["cycles_read_dram"]  += s["cycles_read_dram"]
    acc["cycles_comp_dram"]  += s["cycles_comp_dram"]
    acc["cycles_write_dram"] += s["cycles_write_dram"]
    acc["cycles_read_rram"]  += s["cycles_read_rram"]
    acc["cycles_comp_rram"]  += s["cycles_comp_rram"]
    acc["cycles_write_rram"] += s["cycles_write_rram"]

    acc["energy_read_dram_nj"]  += s["energy_read_dram_nj"]
    acc["energy_comp_dram_nj"]  += s["energy_comp_dram_nj"]
    acc["energy_write_dram_nj"] += s["energy_write_dram_nj"]
    acc["energy_read_rram_nj"]  += s["energy_read_rram_nj"]
    acc["energy_comp_rram_nj"]  += s["energy_comp_rram_nj"]
    acc["energy_write_rram_nj"] += s["energy_write_rram_nj"]

def empty_acc():
    return {
        "cycles": 0,
        "macs": 0,
        "energy_nj": 0.0,
        "breakdown_energy_nj": defaultdict(float),
        "breakdown_macs": defaultdict(float),
        "breakdown_cycles": defaultdict(float),

        "cycles_read_dram":  0,
        "cycles_comp_dram":  0,
        "cycles_write_dram": 0,
        "cycles_read_rram":  0,
        "cycles_comp_rram":  0,
        "cycles_write_rram": 0,

        "energy_read_dram_nj":  0.0,
        "energy_comp_dram_nj":  0.0,
        "energy_write_dram_nj": 0.0,
        "energy_read_rram_nj":  0.0,
        "energy_comp_rram_nj":  0.0,
        "energy_write_rram_nj": 0.0,
    }

def pct(x, total):
    return (x / total * 100.0) if total else 0.0

def main():
    ap = argparse.ArgumentParser(description="Batch-run, expand encoder/decoder layers, and aggregate.")
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--enc-layers", type=int, default=24, help="encoder_attention/ffn 的层数")
    ap.add_argument("--dec-layers", type=int, default=24, help="decoder_attention/ffn 的层数, mobilevlm 1.7B 24layers, 3B 32 layers, fastvlm 0.6B 24layers, 1.7B 28layers")
    ap.add_argument("--stage4_layers", type=int, default=4, help="fastvlm stage4 的层数")
    ap.add_argument("--freq-ghz", type=float, default=1.0)
    args = ap.parse_args()

    # # 固定顺序 mobilevlm
    # stages = [
    #     "patch_embed_B.json",
    #     "encoder_attention_B.json",
    #     "encoder_ffn_B.json",
    #     "connector_B.json",
    #     "decoder_attention_B.json",
    #     "decoder_ffn_B.json",
    # ]

    # # 层数放大（其余默认 1）
    # multipliers = {
    #     "encoder_attention_B.json": args.enc_layers,
    #     "encoder_ffn_B.json": args.enc_layers,
    #     "decoder_attention_B.json": args.dec_layers,
    #     "decoder_ffn_B.json": args.dec_layers,
    # }

    # 固定顺序 fastvlm
    stages = [
        "stem_stage123.json",
        "stage4.json",
        "patch_embed_4_5.json",
        "stage5.json",
        "connector.json",
        "decoder_attention.json",
        "decoder_ffn.json",
    ]

    # 层数放大（其余默认 1）
    multipliers = {
        "stage4.json": args.stage4_layers,
        "decoder_attention.json": args.dec_layers,
        "decoder_ffn.json": args.dec_layers,
    }

    # —— 跑每个环节，按层数放大，保存“环节级”结果（用于环节占比）
    stage_results = {}
    for name in stages:
        path = os.path.join(args.model_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        raw = run_one(path, freq_ghz=args.freq_ghz, default_bits=16)
        k = multipliers.get(name, 1)
        stage_results[name] = scale_stats(raw, k)

    # —— 计算总量（环节加总）
    grand = empty_acc()
    for name in stages:
        add_into(grand, stage_results[name])

    # —— 以算子为维度：聚合（已放大后的环节相加）
    ops = defaultdict(lambda: {"cycles": 0.0, "macs": 0.0, "energy_nj": 0.0})
    for name in stages:
        s = stage_results[name]
        for op, v in s["breakdown_cycles"].items():
            ops[op]["cycles"] += v
        for op, v in s["breakdown_macs"].items():
            ops[op]["macs"] += v
        for op, v in s["breakdown_energy_nj"].items():
            ops[op]["energy_nj"] += v

    # —— 导出：环节层面的统计（含占比）
    import csv
    with open(stage_output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage",
                    "cycles", "cycles_%", 
                    "macs", "macs_%", 
                    "energy_nJ", "energy_%", 
                    "energy_J", "time_s@{:.2f}GHz".format(args.freq_ghz)])
        for name in stages:
            s = stage_results[name]
            time_s = s["cycles"] / (args.freq_ghz * 1e9)
            w.writerow([
                name,
                s["cycles"], pct(s["cycles"], grand["cycles"]),
                s["macs"], pct(s["macs"], grand["macs"]),
                f"{s['energy_nj']:.6f}", pct(s["energy_nj"], grand["energy_nj"]),
                f"{s['energy_nj']*1e-9:.9f}", f"{time_s:.6f}"
            ])

    # —— 导出：算子层面的统计（含占比）
    total_cycles = grand["cycles"]
    total_macs   = grand["macs"]
    total_energy = grand["energy_nj"]

    with open(ops_output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["op_type",
                    "cycles", "cycles_%", 
                    "macs", "macs_%", 
                    "energy_nJ", "energy_%", 
                    "energy_J"])
        for op, v in sorted(ops.items(), key=lambda x: x[1]["energy_nj"], reverse=True):
            w.writerow([
                op,
                int(v["cycles"]), pct(v["cycles"], total_cycles),
                int(v["macs"]), pct(v["macs"], total_macs),
                f"{v['energy_nj']:.6f}", pct(v["energy_nj"], total_energy),
                f"{v['energy_nj']*1e-9:.9f}",
            ])

    # —— 输出总量
    total_time_s = total_cycles / (args.freq_ghz * 1e9)
    with open(total_output_path, "w") as f:
        f.write("==== TOTALS (after layer expansion) ====\n")
        f.write(f"cycles: {total_cycles}\n")
        f.write(f"macs:   {total_macs}\n")
        f.write(f"energy: {total_energy:.6f} nJ  ({total_energy*1e-9:.9f} J)\n")
        f.write(f"time@{args.freq_ghz}GHz: {total_time_s:.6f} s\n")

        # —— 新增：设备×方向×类型（cycles）——
        f.write("\n==== Detailed Cycle Breakdown ====\n")
        f.write(f"cycles_read_dram:    {grand['cycles_read_dram']}\n")
        f.write(f"cycles_comp_dram:    {grand['cycles_comp_dram']}\n")
        f.write(f"cycles_write_dram:   {grand['cycles_write_dram']}\n")
        f.write(f"cycles_read_rram:    {grand['cycles_read_rram']}\n")
        f.write(f"cycles_comp_rram:    {grand['cycles_comp_rram']}\n")
        f.write(f"cycles_write_rram:   {grand['cycles_write_rram']}\n")

        # —— 新增：设备×方向×类型（energy, nJ）——
        f.write("\n==== Detailed Energy Breakdown (nJ) ====\n")
        f.write(f"energy_read_dram:    {grand['energy_read_dram_nj']:.6f}\n")
        f.write(f"energy_comp_dram:    {grand['energy_comp_dram_nj']:.6f}\n")
        f.write(f"energy_write_dram:   {grand['energy_write_dram_nj']:.6f}\n")
        f.write(f"energy_read_rram:    {grand['energy_read_rram_nj']:.6f}\n")
        f.write(f"energy_comp_rram:    {grand['energy_comp_rram_nj']:.6f}\n")
        f.write(f"energy_write_rram:   {grand['energy_write_rram_nj']:.6f}\n")

    print("跑完了, 去 result/ 看")

if __name__ == "__main__":
    main()
