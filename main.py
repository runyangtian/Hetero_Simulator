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
    parser.add_argument('--json', type=str, default='', help='Path to JSON model spec')
    args = parser.parse_args()

    # Device parameters
    # --- 3D DRAM (monolithic/hybrid bonded, research level) ---
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
    compiler = SimpleCompiler(model, rram, dram, tile_K=128, tile_M=128, tile_N=128)
    schedule = compiler.compile()

    # Simulate
    sim = Simulator(model, schedule, rram, dram, dram_cu, rram_cu)
    # sim = Simulator(model, schedule, dram, dram, dram_cu, dram_cu)

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
    
    # ===== cycle breakdown =====
    print('\nDetailed Cycle Breakdown:')
    print(f'  DRAM Read cycles:   {stats.cycles_read_dram}')
    print(f'  DRAM Compute cycles:{stats.cycles_comp_dram}')
    print(f'  DRAM Write cycles:  {stats.cycles_write_dram}')
    print(f'  RRAM Read cycles:   {stats.cycles_read_rram}')
    print(f'  RRAM Compute cycles:{stats.cycles_comp_rram}')
    print(f'  RRAM Write cycles:  {stats.cycles_write_rram}')

    # ===== energy breakdown =====
    print('\nDetailed Energy Breakdown (nJ):')
    print(f'  DRAM Read energy:   {stats.energy_read_dram_nj:.2f}')
    print(f'  DRAM Compute energy:{stats.energy_comp_dram_nj:.2f}')
    print(f'  DRAM Write energy:  {stats.energy_write_dram_nj:.2f}')
    print(f'  RRAM Read energy:   {stats.energy_read_rram_nj:.2f}')
    print(f'  RRAM Compute energy:{stats.energy_comp_rram_nj:.2f}')
    print(f'  RRAM Write energy:  {stats.energy_write_rram_nj:.2f}')



if __name__ == '__main__':
    main()