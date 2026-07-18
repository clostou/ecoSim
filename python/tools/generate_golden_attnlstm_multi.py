#!/usr/bin/env python3
"""
generate_golden_attnlstm_multi.py — 生成多步BPTT golden数据

输出文件：golden_attnlstm_multi.bin

Usage: python generate_golden_attnlstm_multi.py [num_steps] [output_path]
"""

import numpy as np
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_golden_attnlstm import *

def main():
    num_steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'golden_attnlstm_multi.bin'
    print(f"Generating multi-step ({num_steps}) AttnLSTM golden data → {out_path}")

    rng = np.random.RandomState(42)
    ref = AttnLSTMReference(rng)

    # 打包权重（与单步相同）
    weights_packed = pack_weights_col_major(ref)

    # 生成N组输入
    observes = []
    inners = []
    for t in range(num_steps):
        obs = rng.randn(OBS_DIM, OBS_N).astype(np.float32) * 0.5
        inn = rng.randn(INNER_DIM).astype(np.float32) * 0.5
        observes.append(obs)
        inners.append(inn)

    # 上游梯度（最后一步的输出梯度）
    g_act_last   = rng.randn(ACT_DIM).astype(np.float32) * 0.5
    g_value_last = rng.randn(1).astype(np.float32).item() * 0.5

    # ---- 前向N步 ----
    hp = np.zeros(D_H, dtype=np.float32)
    cp = np.zeros(D_H, dtype=np.float32)
    caches = []
    h_outs = []
    c_outs = []
    acts = []
    values = []
    h_states = [hp.copy()]  # h at step 0
    c_states = [cp.copy()]  # c at step 0

    for t in range(num_steps):
        act, val, hn, cn, cache = ref.forward(observes[t], inners[t], hp, cp)
        acts.append(act)
        values.append(val)
        h_outs.append(hn)
        c_outs.append(cn)
        caches.append(cache)
        hp = hn
        cp = cn
        h_states.append(hn.copy())
        c_states.append(cn.copy())

    # ---- BPTT反向（手动链式求导，确保与GPU一致） ----
    ref.zero_grad()

    d_hp_carry = np.zeros(D_H, dtype=np.float32)
    d_cp_carry = np.zeros(D_H, dtype=np.float32)
    g_obs_all_steps = []
    g_hp_all_steps = []
    g_cp_all_steps = []

    for t in reversed(range(num_steps)):
        cache = caches[t]
        observe = observes[t]

        # Only last step gets g_act (no intermediate supervision)
        g_act_t = g_act_last if t == num_steps - 1 else np.zeros(ACT_DIM, dtype=np.float32)
        g_value_t = g_value_last if t == num_steps - 1 else 0.0

        # ---- Critic backward (only for last step) ----
        if t == num_steps - 1:
            g_obs_critic = ref.backward_critic(g_value_t, cache, np.zeros_like(observe))
        else:
            g_obs_critic = np.zeros_like(observe)

        # ---- Actor backward ----
        # Output layer
        d_lstm_o = g_act_t * sigmoid_d(cache['act'])
        ref.grad_Wf += np.outer(d_lstm_o, cache['lstm_o'])
        ref.grad_bf += d_lstm_o

        d_z_o = ref.Wf.T @ d_lstm_o
        d_h_new = d_hp_carry.copy()  # BPTT carry from future steps
        d_h_new[8:16] += d_z_o  # output layer gradient (only for last step)

        # LSTM gate derivatives
        gi = cache['gate_i']; go = cache['gate_o']; ca = cache['c_alt']
        th = cache['tanhc']; c_prev = c_states[t]

        d_gate_o = d_h_new * th * sigmoid_d(go)
        d_c_new  = d_h_new * go * tanh_d(th) + d_cp_carry  # BPTT carry
        d_c_alt  = d_c_new * gi * tanh_d(ca)
        d_gate_i = d_c_new * (ca - c_prev) * sigmoid_d(gi)
        d_c_prev = d_c_new * (1.0 - gi)

        # Accumulate Wico gradients
        x = cache['x']
        ref.grad_Wico[:, :D_IN]          += np.outer(d_gate_i, x)
        ref.grad_Wico[:, D_IN:2*D_IN]    += np.outer(d_c_alt, x)
        ref.grad_Wico[:, 2*D_IN:3*D_IN]  += np.outer(d_gate_o, x)
        ref.grad_bico[:D_H]      += d_gate_i
        ref.grad_bico[D_H:2*D_H] += d_c_alt
        ref.grad_bico[2*D_H:]    += d_gate_o

        # d_x
        Wi = ref.Wico[:, :D_IN]; Wcr = ref.Wico[:, D_IN:2*D_IN]; Wo = ref.Wico[:, 2*D_IN:3*D_IN]
        d_x = Wi.T @ d_gate_i + Wcr.T @ d_c_alt + Wo.T @ d_gate_o

        # Split d_x
        d_hp_x = d_x[:D_H]
        d_ob_flat = d_x[D_H + INNER_DIM:]

        # sigmoid backward: _observe = sigmoid(O_flat + bcat)
        d_pre_sigmoid = d_ob_flat * sigmoid_d(cache['x'][D_H + INNER_DIM:])

        # Attention backward
        dO = d_pre_sigmoid.reshape(H_Q, D_E)
        p = cache['p']; k_arr = cache['k']; v_arr = cache['v']; q_arr = cache['q']
        hp_slice = h_states[t][:D_H1]

        for q_h in range(H_Q):
            kv_h = q_h // Q_N
            # PV backward
            dV_h = np.outer(dO[q_h], p[q_h])
            dP_h = v_arr[kv_h].T @ dO[q_h]
            # Softmax backward
            sum_dp_p = np.sum(dP_h * p[q_h])
            dS_h = p[q_h] * (dP_h - sum_dp_p) / np.sqrt(D_E)
            # QK backward
            dK_h = np.outer(q_arr[q_h], dS_h)
            dQ_h = dS_h @ k_arr[kv_h].T
            # Weight gradients
            ref.grad_Wkv[kv_h, :, :OBS_DIM]  += dK_h @ observe.T
            ref.grad_Wkv[kv_h, :, OBS_DIM:]  += dV_h @ observe.T
            ref.grad_Wq[q_h] += np.outer(dQ_h, hp_slice)
            # Input gradients
            Wk_h = ref.Wkv[kv_h, :, :OBS_DIM]
            Wv_h = ref.Wkv[kv_h, :, OBS_DIM:]
            g_obs_critic += Wk_h.T @ dK_h + Wv_h.T @ dV_h
            d_hp_x[:D_H1] += ref.Wq[q_h].T @ dQ_h

        # bcat gradient
        ref.grad_bcat += d_pre_sigmoid

        # BPTT carry to previous step
        d_hp_carry = d_hp_x
        d_cp_carry = d_c_prev

        g_obs_all_steps.append(g_obs_critic)
        g_hp_all_steps.append(d_hp_x)
        g_cp_all_steps.append(d_c_prev)

    # Pack all gradients
    grads_packed = pack_grads_col_major(ref)

    # ================================================================
    # 写二进制文件
    # ================================================================
    with open(out_path, 'wb') as f:
        f.write(b'ATLM')  # Multi-step magic
        f.write(struct.pack('i', 1))

        # 维度头 + num_steps
        f.write(struct.pack('iiiiiiiiiiiiii',
            OBS_N, OBS_DIM, INNER_DIM, ACT_DIM, D_E, H_KV, H_Q,
            D_H, D_H1, D_H2, D_IN, D_C, WEIGHT_FLOATS, num_steps))

        def write_arr(name, arr):
            data = np.asarray(arr, dtype=np.float32).flatten()
            f.write(struct.pack('i', data.size))
            f.write(data.tobytes())
            print(f"  wrote {name}: {data.size} floats")

        write_arr('weights', weights_packed)

        # Per-step data
        for t in range(num_steps):
            write_arr(f'observe[{t}]', observes[t])
            write_arr(f'inner[{t}]', inners[t])
            write_arr(f'h_prev[{t}]', h_states[t])
            write_arr(f'c_prev[{t}]', c_states[t])

            # Cache data (what GPU backward needs)
            cache = caches[t]
            write_arr(f'x_kv[{t}]', cache['x_kv'])
            # k: [H_KV, D_E, OBS_N] → flatten
            write_arr(f'k[{t}]', cache['k'].flatten())
            write_arr(f'v[{t}]', cache['v'].flatten())
            write_arr(f'q[{t}]', cache['q'].flatten())
            # p: golden is [H_Q, OBS_N]=[4,16], GPU expects [H_Q*H_KV*OBS_N]=[4*2*16]=128
            # Expand: p_gpu[(q_h*H_KV + kv_h)*OBS_N + j] = p_golden[q_h, j]
            p_gpu = np.zeros(H_Q * H_KV * OBS_N, dtype=np.float32)
            for q_h in range(H_Q):
                kv_h = q_h // Q_N
                p_gpu[(q_h * H_KV + kv_h) * OBS_N:(q_h * H_KV + kv_h + 1) * OBS_N] = cache['p'][q_h]
            write_arr(f'p[{t}]', p_gpu)
            write_arr(f'x[{t}]', cache['x'])
            write_arr(f'h_prev_c[{t}]', cache['h_prev'])
            write_arr(f'c_prev_c[{t}]', cache['c_prev'])
            write_arr(f'gate_i[{t}]', cache['gate_i'])
            write_arr(f'gate_o[{t}]', cache['gate_o'])
            write_arr(f'c_alt[{t}]', cache['c_alt'])
            write_arr(f'tanhc[{t}]', cache['tanhc'])
            write_arr(f'lstm_o[{t}]', cache['lstm_o'])
            write_arr(f'act_cache[{t}]', cache['act'])
            write_arr(f'c_h1[{t}]', cache['c_h1'])
            write_arr(f'argmax_c_h1[{t}]', cache['argmax_c_h1'].astype(np.float32).flatten())
            write_arr(f'c_h2[{t}]', cache['c_h2'])
            write_arr(f'c_h3[{t}]', cache['c_h3'])

        # Outputs per step
        for t in range(num_steps):
            write_arr(f'act_out[{t}]', acts[t])
            write_arr(f'value_out[{t}]', np.array([values[t]], dtype=np.float32))
            write_arr(f'h_new[{t}]', h_outs[t])
            write_arr(f'c_new[{t}]', c_outs[t])

        # Upstream gradients
        write_arr('g_act', g_act_last)
        write_arr('g_value', np.array([g_value_last], dtype=np.float32))

        # Accumulated gradients
        write_arr('grads', grads_packed)

        # Per-step input gradients (from BPTT)
        for t in reversed(range(num_steps)):
            idx = num_steps - 1 - t
            write_arr(f'g_observe[{idx}]', g_obs_all_steps[idx])
            write_arr(f'g_h_prev[{idx}]', g_hp_all_steps[idx])
            write_arr(f'g_c_prev[{idx}]', g_cp_all_steps[idx])

    size = Path(out_path).stat().st_size
    print(f"\nTotal: {size} bytes, {num_steps} steps")
    print(f"Step 0 act: [{acts[0][0]:.4f}, {acts[0][1]:.4f}], value: {values[0]:.4f}")
    if num_steps > 1:
        print(f"Step {num_steps-1} act: [{acts[-1][0]:.4f}, {acts[-1][1]:.4f}], value: {values[-1]:.4f}")
    print(f"Grad Wkv max abs: {np.max(np.abs(grads_packed[:256])):.6f}")
    print("Done.")


if __name__ == '__main__':
    main()
