# Simulator for Heterogenous 3D-DRAM 3D-RRAM PIM
how to run: 
1. Modify ./model_json/test.json to describe your model architecture
2. Run: python main.py

TODO:
1. 仿真器并没有考虑偏置项，如 MLP 中 X=W2(W1*X+b1)+b2 的b1，b2
2. 仿真器目前只添加了encoder block的仿真过程

# Simulator 参数表

## 3D DRAM (Monolithic/Hybrid Bonded, Research Level)
| 参数 | 值 | 说明 |
|------|----|------|
| name | 3D_DRAM | 设备名称 |
| capacity_bits | 8 × 1024 × 1024 × 1024 × 8 = **64 Gbit (8 GB)** | 原型容量 |
| read_bw_bits_per_cycle | 4096 | ≈ 0.5 TB/s @ 1 GHz |
| write_bw_bits_per_cycle | 4096 | 假设和读带一样 |
| read_energy_per_bit | 0.0003 pJ/bit | 0.3 pJ/bit |
| write_energy_per_bit | 0.0004 pJ/bit | 写能耗稍高 |
| access_latency_cycles | 20 cycles | 20 ns @ 1 GHz |

---

## 3D RRAM (nvCIM style)
| 参数 | 值 | 说明 |
|------|----|------|
| name | 3D_RRAM | 设备名称 |
| capacity_bits | 8 × 1024 × 1024 × 1024 × 8 = **64 Gbit (8 GB)** | 容量 |
| read_bw_bits_per_cycle | 256 | ≈ 25 GB/s |
| write_bw_bits_per_cycle | 128 | 写带宽更低 |
| read_energy_per_bit | 0.00002 pJ/bit | 20 fJ/bit |
| write_energy_per_bit | 0.00005 pJ/bit | 50 fJ/bit |
| access_latency_cycles | 10 cycles | 10 ns |

---

## Compute Unit (CU)
| 参数 | 值 | 说明 |
|------|----|------|
| macs_per_cycle | 8192 | 每周期可执行的MAC数 |
| energy_per_mac_nj | 0.00015 nJ | 每个MAC能耗 |

---

## Compiler 设置 (SimpleCompiler)
| 参数 | 值 | 说明 |
|------|----|------|
| bits_per_element | 8 | 数据精度（bit/元素） |
| tile_K | 256 | 矩阵分块维度 K |
| tile_M | 128 | 矩阵分块维度 M |
| tile_N | 128 | 矩阵分块维度 N |

---

## Simulator 全局参数
| 参数 | 值 | 说明 |
|------|----|------|
| bits_per_element | 8 | 默认元素位宽 |
| ucie_bandwidth | 4e12 bit/cycle = **4 Tbps** | UCIe PHY 带宽 (16G × 64 lanes × 4 modules) |
| ucie_energy_per_bit | 0.5 pJ/bit | UCIe 传输能耗 |
