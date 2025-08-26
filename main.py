# 整合所有模块并启动模拟

import argparse
import json
import time

from hardware_models import MemoryDevice, ComputeUnit
from model import Model
from loader import JSONModelLoader
from compiler import SimpleCompiler
from simulator import Simulator

def main():
    parser = argparse.ArgumentParser(description='3D Hybrid Memory Simulator (JSON-driven)')
    parser.add_argument('--json', type=str, default='./model_json/patch_embed.json', help='Path to JSON model spec')
    args = parser.parse_args()

    # Device parameters
    # --- 3D DRAM (monolithic/hybrid bonded, research level) ---
    dram = MemoryDevice(
        name='3D_DRAM',
        capacity_bits = int(6.25*1024*1024*1024*8),    # 6.25GB
        read_bw_bits_per_cycle  = 1024,        
        write_bw_bits_per_cycle = 1024,    
        read_energy_per_bit  = 0.000429,         # 0.429 pJ/bit
        write_energy_per_bit = 0.000429,         
        read_latency_cycles = 3,
        write_latency_cycles = 3             
    )

    # --- 3D RRAM (nvCIM style) ---
    rram = MemoryDevice(
        name='3D_RRAM',
        capacity_bits = int(2*1024*1024*1024*8),        # 2GB
        read_bw_bits_per_cycle  = 768,        # interface BW=12.3 TB/s     12.3 *8 * 1024  =  
        write_bw_bits_per_cycle = 768,        
        read_energy_per_bit  = 0.0004,         # nJ/bit
        write_energy_per_bit = 0.00133,        # nJ/bit
        read_latency_cycles = 3,
        write_latency_cycles = 11            
    )

    # cu = ComputeUnit(macs_per_cycle=8192, energy_per_mac_nj=0.00015)

    # --- DRAM-CU：MAC + SFE ---
    dram_cu = ComputeUnit(
        name='DRAM_CU',
        macs_per_cycle=64*16*16,      # 8×8 MAC/PE × 16 PE/PU × 16 PU = 16384 MAC/cycle
        energy_per_mac_nj=0.000268,   # 由 32 TFLOPS & 4.3 W 反推 ≈ 0.268 pJ/MAC
        sfe_ops_per_cycle=256*16,      # Special Func. Engine: 256-way SIMD（近似設為每 PU 256 op/cycle × 16 PU）
        sfe_energy_per_op_nj=0.00005, # SFE 單元素能耗（保守地遠低於 MAC）
    )

    # --- RRAM-CU：只有 MAC（無 SFE）---
    rram_cu = ComputeUnit(
        name='RRAM_CU',
        macs_per_cycle=25*16*16,           # 5×5 MAC/PE × 16 PE/PU × 16 PU = 6400 MAC/cycle
        energy_per_mac_nj=0.000268,    # 由 12.5 TFLOPS & 1.68 W 反推 ~ 0.268 pJ/MAC（與 DRAM 類似量級）
        sfe_ops_per_cycle=256*16,      
        sfe_energy_per_op_nj=0.00005, # 为了仿真给rram也加上SFE
    )


    # Load model
    if args.json:
        with open(args.json, 'r') as f:
            spec = json.load(f)
        loader = JSONModelLoader(default_bits=16)
        model = loader.build(spec)
    else:
        print("No JSON file provided.")

    # Compile
    # compiler = SimpleCompiler(model, rram, dram, cu, bits_per_element=16, tile_K=256, tile_M=128, tile_N=128)
    compiler = SimpleCompiler(model, rram, dram, bits_per_element=16, tile_K=256, tile_M=128, tile_N=128)
    schedule = compiler.compile()

    # Simulate
    # sim = Simulator(model, schedule, rram, dram, cu, bits_per_element=16)
    sim = Simulator(model, schedule, rram, dram, dram_cu, rram_cu, bits_per_element=16)

    stats = sim.run()

    # Print results
    print("\nSimulation result (JSON-driven graph on hetero PIM):")
    print(f"Total cycles: {stats.cycles}")
    print(f"Total MACs: {stats.macs}") 
    # print(f"Total energy (nJ): {stats.energy_nj:.2f}")
    
    freq_ghz = 1.0
    exec_time_s = stats.cycles / (freq_ghz * 1e9)
    energy_j = stats.energy_nj * 1e-9
    print(f"Estimated wall time @1GHz: {exec_time_s:.6f} s")
    print(f"Total energy (J): {energy_j:.6f} J")
    
    print('\nEnergy Breakdown (nJ):')
    for k,v in stats.breakdown.items():
        print(f'  {k}: {v:.2f}')

    print('\nMAC Breakdown:')
    for k,v in stats.macs_breakdown.items():
        print(f'  {k}: {v}')

    print('\nCycle Breakdown:')
    for k,v in stats.cycles_breakdown.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()