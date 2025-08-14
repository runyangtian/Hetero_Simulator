# 整合所有模块并启动模拟

import argparse
import json
import time

from hardware_models import MemoryDevice, ComputeUnit, ACU
from model import Model
from loader import JSONModelLoader
from compiler import SimpleCompiler
from simulator import Simulator

def main():
    parser = argparse.ArgumentParser(description='3D Hybrid Memory Simulator (JSON-driven)')
    parser.add_argument('--json', type=str, default='./model_json/test.json', help='Path to JSON model spec')
    args = parser.parse_args()

    # Device parameters
    rram = MemoryDevice(name='3D_RRAM', capacity_bits=32*1024*1024*8, read_bw_bits_per_cycle=1024*8, write_bw_bits_per_cycle=512*8,
                        read_energy_per_bit=0.0005/8, write_energy_per_bit=0.0008/8, access_latency_cycles=5)
    dram = MemoryDevice(name='3D_DRAM', capacity_bits=256*1024*1024*8, read_bw_bits_per_cycle=8192*8, write_bw_bits_per_cycle=4096*8,
                        read_energy_per_bit=0.002/8, write_energy_per_bit=0.003/8, access_latency_cycles=50)
    cu = ComputeUnit(macs_per_cycle=8192, energy_per_mac_nj=0.00015)
    acu = ACU(throughput_elements_per_cycle=256, energy_per_element_nj=0.0005, call_latency_cycles=2)

    # Load model
    if args.json:
        with open(args.json, 'r') as f:
            spec = json.load(f)
        loader = JSONModelLoader(default_bits=16)
        model = loader.build(spec)
    else:
        print("No JSON file provided.")

    # Compile
    compiler = SimpleCompiler(model, rram, dram, cu, bits_per_element=16, tile_K=256, tile_M=128, tile_N=128)
    schedule = compiler.compile()

    # Simulate
    sim = Simulator(model, schedule, rram, dram, cu, acu, bits_per_element=16)
    
    t0 = time.time()
    stats = sim.run()
    t1 = time.time()
    
    # Print results
    print("\nSimulation result (JSON-driven graph on hetero PIM + ACU):")
    print(f"Total cycles: {stats.cycles}")
    print(f"Total MACs: {stats.macs}")
    print(f"Total energy (nJ): {stats.energy_nj:.2f}")
    
    freq_ghz = 1.0
    exec_time_s = stats.cycles / (freq_ghz * 1e9)
    energy_j = stats.energy_nj * 1e-9
    print(f"Estimated wall time @1GHz: {exec_time_s:.6f} s")
    print(f"Estimated energy (J): {energy_j:.6f} J")
    
    print('\nBreakdown (Energy nJ):')
    for k,v in stats.breakdown.items():
        print(f'  {k}: {v:.2f}')

    print(f"\nSimulation runtime: {t1-t0:.3f}s")


if __name__ == '__main__':
    main()