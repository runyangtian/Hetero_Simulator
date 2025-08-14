# Simulator for Heterogenous 3D-DRAM 3D-RRAM PIM
how to run: 
1. Modify ./model_json/test.json to describe your model architecture
2. Run: python main.py

TODO:
1. 仿真器并没有考虑偏置项，如 MLP 中 X=W2(W1*X+b1)+b2 的b1，b2
2. 仿真器目前只添加了encoder block的仿真过程