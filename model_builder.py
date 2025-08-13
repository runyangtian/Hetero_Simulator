from model import Model
from operations import LayerNorm, MatMul, Attention, BinaryOp, UnaryOp

def build_transformer_encoder(num_layers=6, seq_len=196, embed_dim=768, bits_act=16, bits_weight=16) -> Model:
    """
    Constructs a Model with `num_layers` of a standard Transformer encoder.
    Layer structure: norm -> attention -> add -> norm -> ffn -> add
    """
    model = Model()
    B = 1  # Batch size
    L = seq_len
    D = embed_dim
    
    # Input tensor
    model.add_tensor('input_tokens', (B, L, D), bits_per_element=bits_act, device='dram')
    last_output = 'input_tokens'

    for l in range(num_layers):
        # --- Attention Block ---
        # 1. LayerNorm
        norm1_in = last_output
        norm1_out = f'norm1_out_{l}'
        model.add_tensor(norm1_out, (B, L, D), bits_act, 'dram')
        model.add_op(LayerNorm(norm1_in, norm1_out))

        # 2. QKV Projections (as MatMuls)
        # Note: In a real scenario, inputs would be reshaped. We simplify this.
        # Assuming input (B, L, D) is treated as (B*L, D) for MatMul
        q_proj_out, k_proj_out, v_proj_out = f'Q_{l}', f'K_{l}', f'V_{l}'
        wq, wk, wv = f'W_q_{l}', f'W_k_{l}', f'W_v_{l}'
        
        model.add_tensor(wq, (D, D), bits_weight, 'rram')
        model.add_tensor(wk, (D, D), bits_weight, 'rram')
        model.add_tensor(wv, (D, D), bits_weight, 'rram')

        model.add_tensor(q_proj_out, (B, L, D), bits_act, 'dram')
        model.add_tensor(k_proj_out, (B, L, D), bits_act, 'dram')
        model.add_tensor(v_proj_out, (B, L, D), bits_act, 'dram')
        
        # We use norm1_out as input to the QKV projections.
        model.add_op(MatMul(norm1_out, wq, q_proj_out))
        model.add_op(MatMul(norm1_out, wk, k_proj_out))
        model.add_op(MatMul(norm1_out, wv, v_proj_out))

        # 3. Attention
        attn_out = f'attn_out_{l}'
        model.add_tensor(attn_out, (B, L, D), bits_act, 'dram')
        model.add_op(Attention(q_proj_out, k_proj_out, v_proj_out, attn_out))

        # 4. Residual Add
        resid1_out = f'resid1_{l}'
        model.add_tensor(resid1_out, (B, L, D), bits_act, 'dram')
        model.add_op(BinaryOp('ADD', attn_out, norm1_in, resid1_out))

        # --- FFN Block ---
        # 5. LayerNorm
        norm2_in = resid1_out
        norm2_out = f'norm2_out_{l}'
        model.add_tensor(norm2_out, (B, L, D), bits_act, 'dram')
        model.add_op(LayerNorm(norm2_in, norm2_out))

        # 6. FFN Layers
        ff1_out = f'ff1_{l}'
        ff1_act_out = f'ff1_act_{l}'
        ff2_out = f'ff2_{l}'
        w1, w2 = f'W1_{l}', f'W2_{l}'
        
        model.add_tensor(w1, (D, D * 4), bits_weight, 'rram')
        model.add_tensor(w2, (D * 4, D), bits_weight, 'rram')
        
        model.add_tensor(ff1_out, (B, L, D * 4), bits_act, 'dram')
        model.add_tensor(ff1_act_out, (B, L, D * 4), bits_act, 'dram')
        model.add_tensor(ff2_out, (B, L, D), bits_act, 'dram')
        
        model.add_op(MatMul(norm2_out, w1, ff1_out))
        model.add_op(UnaryOp('GELU', ff1_out, ff1_act_out))
        model.add_op(MatMul(ff1_act_out, w2, ff2_out))

        # 7. Residual Add
        resid2_out = f'resid2_{l}'
        model.add_tensor(resid2_out, (B, L, D), bits_act, 'dram')
        model.add_op(BinaryOp('ADD', ff2_out, norm2_in, resid2_out))

        last_output = resid2_out
        
    return model