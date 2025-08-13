# FILE: main.py
# -------------
# Main entry point for the simulation. Sets up hardware, builds the model,
# runs the compiler and simulator, and prints the results.

import time
import sys

# Add path to other modules if they are in a different directory
# sys.path.append('path/to/your/modules') 

from datatypes import MemoryDevice, ComputeUnit, ACU
from model_builder import build_transformer_encoder
from compiler import SimpleCompiler
from simulator import Simulator

if __name__ == '__main__':
    # 1. Define Hardware Parameters
    # Parameters are converted to bit-level units for consistency
    rram = MemoryDevice(
        name='3D_RRAM',
        capacity_bits=32 * 1024 * 1024 * 8,       # 32 MB
        read_bw_bits_per_cycle=1024 * 8,          # 1KB/cycle
        write_bw_bits_per_cycle=512 * 8,          # 0.5KB/cycle
        read_energy_per_bit=0.0005 / 8,           # nJ/bit
        write_energy_per_bit=0.0008 / 8,          # nJ/bit
        access_latency_cycles=5
    )
    dram = MemoryDevice(
        name='3D_DRAM',
        capacity_bits=256 * 1024 * 1024 * 8,      # 256 MB
        read_bw_bits_per_cycle=8192 * 8,          # 8KB/cycle
        write_bw_bits_per_cycle=4096 * 8,         # 4KB/cycle
        read_energy_per_bit=0.002 / 8,            # nJ/bit
        write_energy_per_bit=0.003 / 8,           # nJ/bit
        access_latency_cycles=50
    )
    cu = ComputeUnit(
        macs_per_cycle=8192,
        energy_per_mac_nj=0.00015
    )
    acu = ACU(
        throughput_elements_per_cycle=256,
        energy_per_element_nj=0.0005,
        call_latency_cycles=2
    )
    
    # 2. Build the Model
    print("Building model...")
    # Using low-precision weights (4-bit) and activations (16-bit)
    model = build_transformer_encoder(
        num_layers=6,
        seq_len=196,
        embed_dim=768,
        bits_act=16,
        bits_weight=4
    )

    # 3. Compile the Model
    print("Compiling schedule...")
    # The compiler needs the bit precision of activations for tiling calculations
    compiler = SimpleCompiler(
        model, rram, dram, cu,
        bits_per_element=16,
        tile_K=256, tile_M=128, tile_N=128
    )
    schedule = compiler.compile()

    # 4. Run the Simulator
    print("Starting simulation...")
    sim = Simulator(
        model, schedule, rram, dram, cu, acu,
        bits_per_element=16
    )

    t0 = time.time()
    stats = sim.run()
    t1 = time.time()

    # 5. Print Results
    print("\n--- Simulation Complete ---")
    print("Configuration: 6-layer Transformer encoder on a PIM+ACU architecture.")
    print(f"Total Cycles: {stats.cycles:,}")
    print(f"Total MACs: {stats.macs:,}")
    print(f"Total Energy: {stats.energy_nj:,.2f} nJ")
    print(f"Total Bits Read: {stats.bits_read:,}")
    print(f"Total Bits Written: {stats.bits_written:,}")
    print(f"Simulation Wall Time: {t1 - t0:.3f} s")

    # Performance Estimates
    freq_ghz = 1.0  # Assuming 1 GHz clock frequency
    exec_time_s = stats.cycles / (freq_ghz * 1e9)
    energy_j = stats.energy_nj * 1e-9
    print(f"\nEstimated Execution Time @ {freq_ghz}GHz: {exec_time_s * 1e3:.4f} ms")
    print(f"Estimated Total Energy: {energy_j * 1e6:,.2f} uJ")

    # Energy Breakdown
    print("\nEnergy Breakdown (nJ):")
    sorted_breakdown = sorted(stats.breakdown.items(), key=lambda item: item[1], reverse=True)
    for k, v in sorted_breakdown:
        print(f"  - {k.capitalize():<12}: {v:,.2f} nJ ({(v / stats.energy_nj) * 100:.1f}%)")