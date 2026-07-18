#!/usr/bin/env python3
"""
generate_golden_attnlstm.py — 为 AttnLSTM CUDA kernel 生成参考前向/反向数据

生成的数据布局与 C++ AttnLstmWeights / AttnLstmCache 完全一致：
- 权重：列优先（column-major）
- 缓存：行优先（row-major）
- LSTM：耦合输入/遗忘门
- 注意力：GQA with h_kv=2, h_q=4，K/V无偏置，bcat仅输出偏置
- U折叠：LSTM递归权重 U_* 折叠进 Wico

输出文件：golden_attnlstm.bin

Usage: python generate_golden_attnlstm.py [output_path]
"""

import numpy as np
import struct
import sys
from pathlib import Path


# ============================================================
# 维度配置（必须与 net_config.cuh DefaultConfig 一致）
# ============================================================
OBS_N     = 16   # M: 观测实体数
OBS_DIM   = 8    # D_obs: 单实体观测特征维
INNER_DIM = 8    # D_inn: 内部状态维
ACT_DIM   = 2    # d_o: 动作维
D_E       = 8    # d_e: 注意力嵌入维
H_KV      = 2    # h_kv: KV头数
H_Q       = 4    # h_q: Q头总数
Q_N       = H_Q // H_KV  # 每KV头的查询数 = 2
D_H       = 16   # d_h: LSTM隐/细胞状态维
D_H1      = 12   # d_h1: Q投影输入维 (h_prev[0:12])
D_H2      = 8    # d_h2: LSTM输出维 (h_new[8:16])
D_IN      = H_Q * D_E + INNER_DIM + D_H  # LSTM输入维 = 32+8+16 = 56
D_C       = 16   # d_c: Critic隐藏维

# 权重区字段大小
WKv_SIZE  = 2 * H_KV * D_E * OBS_DIM   # 256
WQ_SIZE   = H_Q * D_E * D_H1           # 384
BCAT_SIZE = H_Q * D_E                  # 32
WICO_SIZE = 3 * D_H * D_IN             # 2688
BICO_SIZE = 3 * D_H                    # 48
WF_SIZE   = D_H2 * ACT_DIM             # 16
BF_SIZE   = ACT_DIM                    # 2
WC1_SIZE  = D_C * OBS_DIM              # 128
BC1_SIZE  = D_C                        # 16
WC2_SIZE  = D_C * (D_C + INNER_DIM)    # 384
BC2_SIZE  = D_C                        # 16
WC3_SIZE  = D_C * D_C                  # 256
BC3_SIZE  = D_C                        # 16
WC4_SIZE  = 1 * D_C                    # 16
BC4_SIZE  = 1                          # 1

# 所有权重字段的总float数
WEIGHT_FLOATS = (
    WKv_SIZE + WQ_SIZE + BCAT_SIZE +
    WICO_SIZE + BICO_SIZE + WF_SIZE + BF_SIZE +
    WC1_SIZE + BC1_SIZE + WC2_SIZE + BC2_SIZE +
    WC3_SIZE + BC3_SIZE + WC4_SIZE + BC4_SIZE
)

# 梯度字段与权重字段一一对应
GRAD_FLOATS = WEIGHT_FLOATS
TOTAL_FLOATS = WEIGHT_FLOATS + GRAD_FLOATS


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_d(y):
    return y * (1.0 - y)


def tanh_d(y):
    return 1.0 - y * y


class AttnLSTMReference:
    """AttnLSTM参考实现，匹配C++布局"""

    def __init__(self, rng: np.random.RandomState):
        # ---- Actor权重（列优先存储，但NumPy中用行优先表示，输出时转置） ----
        # Wkv [H_KV, D_E, 2*OBS_DIM] 列优先
        self.Wkv = rng.randn(H_KV, D_E, 2 * OBS_DIM).astype(np.float32) / np.sqrt(OBS_DIM)
        # Wq [H_Q, D_E, D_H1] 列优先
        self.Wq = rng.randn(H_Q, D_E, D_H1).astype(np.float32) / np.sqrt(D_H1)
        self.bcat = rng.randn(H_Q * D_E).astype(np.float32) * 0.01
        # Wico [D_H, 3*D_IN] 列优先（[Wi | Wc | Wo]沿D_IN维拼接）
        self.Wico = rng.randn(D_H, 3 * D_IN).astype(np.float32) / np.sqrt(D_IN)
        self.bico = rng.randn(3 * D_H).astype(np.float32) * 0.01
        # Wf [ACT_DIM, D_H2] 列优先
        self.Wf = rng.randn(ACT_DIM, D_H2).astype(np.float32) / np.sqrt(D_H2)
        self.bf = rng.randn(ACT_DIM).astype(np.float32) * 0.01

        # ---- Critic权重（列优先） ----
        self.Wc1 = rng.randn(D_C, OBS_DIM).astype(np.float32) / np.sqrt(OBS_DIM)
        self.bc1 = rng.randn(D_C).astype(np.float32) * 0.01
        self.Wc2 = rng.randn(D_C, D_C + INNER_DIM).astype(np.float32) / np.sqrt(D_C + INNER_DIM)
        self.bc2 = rng.randn(D_C).astype(np.float32) * 0.01
        self.Wc3 = rng.randn(D_C, D_C).astype(np.float32) / np.sqrt(D_C)
        self.bc3 = rng.randn(D_C).astype(np.float32) * 0.01
        self.Wc4 = rng.randn(1, D_C).astype(np.float32) / np.sqrt(D_C)
        self.bc4 = rng.randn(1).astype(np.float32) * 0.01

        self.zero_grad()

    def zero_grad(self):
        self.grad_Wkv  = np.zeros_like(self.Wkv)
        self.grad_Wq   = np.zeros_like(self.Wq)
        self.grad_bcat = np.zeros_like(self.bcat)
        self.grad_Wico = np.zeros_like(self.Wico)
        self.grad_bico = np.zeros_like(self.bico)
        self.grad_Wf   = np.zeros_like(self.Wf)
        self.grad_bf   = np.zeros_like(self.bf)
        self.grad_Wc1  = np.zeros_like(self.Wc1)
        self.grad_bc1  = np.zeros_like(self.bc1)
        self.grad_Wc2  = np.zeros_like(self.Wc2)
        self.grad_bc2  = np.zeros_like(self.bc2)
        self.grad_Wc3  = np.zeros_like(self.Wc3)
        self.grad_bc3  = np.zeros_like(self.bc3)
        self.grad_Wc4  = np.zeros_like(self.Wc4)
        self.grad_bc4  = np.zeros_like(self.bc4)

    def forward(self, observe, inner, h_prev, c_prev):
        """
        单步前向，返回 (act, value, h_new, c_new, cache_dict)
        observe: [OBS_DIM, OBS_N]
        inner:   [INNER_DIM]
        h_prev:  [D_H]
        c_prev:  [D_H]
        """
        cache = {}
        cache['x_kv'] = observe.copy()

        # ---- K/V投影 ----
        # Wkv [H_KV, D_E, 2*OBS_DIM]: 末维 [Wk|Wv]
        Wk = self.Wkv[:, :, :OBS_DIM]  # [H_KV, D_E, OBS_DIM]
        Wv = self.Wkv[:, :, OBS_DIM:]  # [H_KV, D_E, OBS_DIM]

        # k[h,d_e,j] = sum_i Wk[h,d_e,i] * observe[i,j]
        k = np.einsum('hei,ij->hej', Wk, observe)  # [H_KV, D_E, OBS_N]
        v = np.einsum('hei,ij->hej', Wv, observe)

        cache['k'] = k.copy()
        cache['v'] = v.copy()

        # ---- Q投影 ----
        # Wq [H_Q, D_E, D_H1]
        # h_prev_slice [D_H1] = h_prev[0:12]
        h_slice = h_prev[:D_H1]
        q = np.einsum('qei,i->qe', self.Wq, h_slice)  # [H_Q, D_E]
        cache['q'] = q.copy()

        # ---- SDPA ----
        # s[q_h, kv_h, j] = sum_e q[q_h, e] * k[kv_h, e, j] / sqrt(D_E)
        # 注意：每个Q头映射到对应的KV头：kv_h = q_h // Q_N
        s = np.zeros((H_Q, OBS_N), dtype=np.float32)
        p = np.zeros((H_Q, OBS_N), dtype=np.float32)
        o = np.zeros((H_Q, D_E), dtype=np.float32)

        for q_h in range(H_Q):
            kv_h = q_h // Q_N
            # s[q_h, j]
            s_h = np.einsum('e,ej->j', q[q_h], k[kv_h]) / np.sqrt(D_E)
            s[q_h] = s_h
            # softmax
            s_max = np.max(s_h)
            p_h = np.exp(s_h - s_max) / np.sum(np.exp(s_h - s_max))
            p[q_h] = p_h
            # o[q_h, e] = sum_j p_h[j] * v[kv_h, e, j]
            o[q_h] = np.einsum('j,ej->e', p_h, v[kv_h])

        # _observe = sigmoid(o.flatten() + bcat)
        _observe = sigmoid(o.flatten() + self.bcat)

        cache['p'] = p.copy()

        # ---- 拼接 LSTM 输入 ----
        x = np.concatenate([h_prev, inner, _observe])  # [D_IN=56]
        cache['x'] = x.copy()
        cache['h_prev'] = h_prev.copy()
        cache['c_prev'] = c_prev.copy()

        # ---- LSTM 三门 ----
        # Wico [D_H, 3*D_IN] = [Wi | Wc | Wo]
        Wi = self.Wico[:, :D_IN]           # [D_H, D_IN]
        Wc = self.Wico[:, D_IN:2*D_IN]     # [D_H, D_IN]
        Wo = self.Wico[:, 2*D_IN:3*D_IN]   # [D_H, D_IN]

        bi = self.bico[:D_H]
        bc = self.bico[D_H:2*D_H]
        bo = self.bico[2*D_H:3*D_H]

        z_i = Wi @ x + bi
        z_c = Wc @ x + bc
        z_o = Wo @ x + bo

        gate_i = sigmoid(z_i)
        c_alt  = np.tanh(z_c)
        gate_o = sigmoid(z_o)

        cache['gate_i'] = gate_i.copy()
        cache['c_alt']  = c_alt.copy()
        cache['gate_o'] = gate_o.copy()

        # 细胞状态 + 隐状态更新
        c_new = (1.0 - gate_i) * c_prev + gate_i * c_alt
        tanhc = np.tanh(c_new)
        h_new = gate_o * tanhc

        cache['tanhc'] = tanhc.copy()

        # ---- 输出层 ----
        lstm_o = h_new[D_H2:]  # [D_H2] = h_new[8:16]
        cache['lstm_o'] = lstm_o.copy()

        # Wf [ACT_DIM, D_H2]
        z_f = self.Wf @ lstm_o + self.bf
        act = sigmoid(z_f)
        cache['act'] = act.copy()

        # ---- Critic 前向 ----
        # L1
        z_c1 = self.Wc1 @ observe + self.bc1[:, np.newaxis]  # [D_C, OBS_N]
        a_c1 = sigmoid(z_c1)
        c_h1_pooled = np.max(a_c1, axis=1)       # [D_C]
        argmax_c_h1 = np.argmax(a_c1, axis=1)    # [D_C]
        cache['c_h1'] = c_h1_pooled.copy()
        cache['argmax_c_h1'] = argmax_c_h1.copy()
        # 缓存用于反向的原始激活（argmax位置的值≈pooled值）
        cache['a_c1_raw'] = a_c1.copy()

        # L2: [c_h1 | inner] → c_h2
        c12_in = np.concatenate([c_h1_pooled, inner])
        z_c2 = self.Wc2 @ c12_in + self.bc2
        c_h2 = sigmoid(z_c2)
        cache['c_h2'] = c_h2.copy()

        # L3
        z_c3 = self.Wc3 @ c_h2 + self.bc3
        c_h3 = sigmoid(z_c3)
        cache['c_h3'] = c_h3.copy()

        # L4
        value = (self.Wc4 @ c_h3 + self.bc4).item()
        cache['value'] = value

        return act, value, h_new, c_new, cache

    def backward_actor(self, g_act, cache):
        """单个cache的Actor反向，返回 (g_observe_actor, g_h_prev, g_c_prev)"""
        observe = cache['x_kv']
        x = cache['x']
        h_prev = cache['h_prev']
        c_prev = cache['c_prev']
        q = cache['q']
        k = cache['k']
        v = cache['v']
        p = cache['p']
        gate_i = cache['gate_i']
        gate_o = cache['gate_o']
        c_alt  = cache['c_alt']
        tanhc  = cache['tanhc']
        lstm_o = cache['lstm_o']

        # ---- R1-R3: 输出层反向 ----
        d_lstm_o = g_act * sigmoid_d(cache['act'])  # [ACT_DIM]
        self.grad_Wf += np.outer(d_lstm_o, lstm_o)   # [ACT_DIM, D_H2]
        self.grad_bf += d_lstm_o

        d_z_o = self.Wf.T @ d_lstm_o  # [D_H2]
        d_h_new = np.zeros(D_H, dtype=np.float32)
        d_h_new[D_H2:] = d_z_o  # scatter到[8:16]

        # ---- R4: 耦合门导数 ----
        d_gate_o = d_h_new * tanhc * sigmoid_d(gate_o)
        d_c_new  = d_h_new * gate_o * tanh_d(tanhc)
        d_c_alt  = d_c_new * gate_i * tanh_d(c_alt)
        d_gate_i = d_c_new * (c_alt - c_prev) * sigmoid_d(gate_i)
        d_c_prev = d_c_new * (1.0 - gate_i)

        # ---- R5: 三门权重梯度 ----
        # grad_Wico [D_H, 3*D_IN]：[Wi | Wc | Wo] 沿列拼接
        self.grad_Wico[:, :D_IN]            += np.outer(d_gate_i, x)
        self.grad_Wico[:, D_IN:2*D_IN]      += np.outer(d_c_alt, x)
        self.grad_Wico[:, 2*D_IN:3*D_IN]    += np.outer(d_gate_o, x)
        self.grad_bico[:D_H]      += d_gate_i
        self.grad_bico[D_H:2*D_H] += d_c_alt
        self.grad_bico[2*D_H:]    += d_gate_o

        # ---- R6: d_x ----
        Wi = self.Wico[:, :D_IN]
        Wc = self.Wico[:, D_IN:2*D_IN]
        Wo = self.Wico[:, 2*D_IN:3*D_IN]
        d_x = Wi.T @ d_gate_i + Wc.T @ d_c_alt + Wo.T @ d_gate_o

        # ---- R7: 拆分d_x ----
        d_h_prev_x = d_x[:D_H]
        d_observe_flat = d_x[D_H + INNER_DIM:]  # [H_Q * D_E]  (grad w.r.t. post-sigmoid _observe)

        # ---- R7b: sigmoid反向 (_observe = sigmoid(O_flat + bcat)) ----
        _observe_post = x[D_H + INNER_DIM:]  # post-sigmoid value from forward cache
        d_pre_sigmoid = d_observe_flat * sigmoid_d(_observe_post)  # d_pre = d_post * σ'(post)

        # ---- R8-R15: 注意力反向 ----
        # d_pre_sigmoid → dO [H_Q, D_E]
        dO = d_pre_sigmoid.reshape(H_Q, D_E)

        g_observe_actor = np.zeros_like(observe)
        d_h_prev_Q = np.zeros(D_H, dtype=np.float32)

        for q_h in range(H_Q):
            kv_h = q_h // Q_N

            # PV backward: dV[e,j] = P[j] * dO[e]
            dV_h = np.outer(dO[q_h], p[q_h])  # [D_E, OBS_N]
            # dP[j] = sum_e dO[e] * v[e, j]
            dP_h = v[kv_h].T @ dO[q_h]         # [OBS_N]

            # Softmax backward
            sum_dp_p = np.sum(dP_h * p[q_h])
            dS_h = p[q_h] * (dP_h - sum_dp_p) / np.sqrt(D_E)

            # QK backward
            dK_h = np.outer(q[q_h], dS_h)      # [D_E, OBS_N]
            dQ_h = dS_h @ k[kv_h].T            # [D_E]

            # K/V权重梯度
            # Wk[kv_h, d_e, i] += dK_h[d_e, j] * observe[i, j]
            self.grad_Wkv[kv_h, :, :OBS_DIM]  += dK_h @ observe.T
            self.grad_Wkv[kv_h, :, OBS_DIM:]  += dV_h @ observe.T

            # Q权重梯度
            self.grad_Wq[q_h] += np.outer(dQ_h, h_prev[:D_H1])

            # 输入梯度
            Wk_h = self.Wkv[kv_h, :, :OBS_DIM]   # [D_E, OBS_DIM]
            Wv_h = self.Wkv[kv_h, :, OBS_DIM:]   # [D_E, OBS_DIM]
            g_observe_actor += Wk_h.T @ dK_h + Wv_h.T @ dV_h

            d_h_prev_Q[:D_H1] += self.Wq[q_h].T @ dQ_h

        # bcat梯度（线性偏置）
        self.grad_bcat += d_pre_sigmoid

        # h_prev总梯度
        g_h_prev = d_h_prev_x + d_h_prev_Q

        return g_observe_actor, g_h_prev, d_c_prev

    def backward_critic(self, g_value, cache, g_observe_from_actor):
        """Critic反向，累加权重梯度，返回g_observe_critic"""
        c_h1_pooled = cache['c_h1']
        c_h2 = cache['c_h2']
        c_h3 = cache['c_h3']
        a_c1_raw = cache['a_c1_raw']
        argmax_c_h1 = cache['argmax_c_h1']
        inner = cache['x'][D_H:D_H + INNER_DIM]

        # L4 backward
        d_c_h3 = self.Wc4.flatten() * g_value  # [D_C]
        self.grad_Wc4 += g_value * c_h3.reshape(1, D_C)
        self.grad_bc4 += g_value

        # L3 backward
        d_c_h3_raw = d_c_h3 * sigmoid_d(c_h3)
        self.grad_Wc3 += np.outer(d_c_h3_raw, c_h2)
        self.grad_bc3 += d_c_h3_raw
        d_c_h2 = self.Wc3.T @ d_c_h3_raw

        # L2 backward
        d_c_h2_raw = d_c_h2 * sigmoid_d(c_h2)
        c12_in = np.concatenate([c_h1_pooled, inner])
        self.grad_Wc2 += np.outer(d_c_h2_raw, c12_in)
        self.grad_bc2 += d_c_h2_raw
        d_c12 = self.Wc2.T @ d_c_h2_raw
        d_c_h1_pooled = d_c12[:D_C]

        # L1 max-pool backward: 梯度只传给argmax位置
        d_c_h1_raw = np.zeros_like(a_c1_raw)
        for r in range(D_C):
            d_c_h1_raw[r, argmax_c_h1[r]] = d_c_h1_pooled[r]

        # L1 sigmoid backward (用原始激活值)
        d_c_h1_raw_sig = d_c_h1_raw * sigmoid_d(a_c1_raw)

        self.grad_Wc1 += d_c_h1_raw_sig @ cache['x_kv'].T
        self.grad_bc1 += np.sum(d_c_h1_raw_sig, axis=1)

        g_observe_critic = self.Wc1.T @ d_c_h1_raw_sig

        # 合并Actor和Critic对observe的梯度
        g_observe = g_observe_from_actor + g_observe_critic

        return g_observe


def pack_weights_col_major(ref):
    """将所有权重以列优先（Fortran order）展平，匹配C++ AttnLstmWeights布局"""

    def col_major(arr):
        """将NumPy行优先数组转置并展平为列优先"""
        return np.asfortranarray(arr).flatten('F')

    packed = np.zeros(WEIGHT_FLOATS, dtype=np.float32)
    offset = 0

    # Wkv [H_KV, D_E, 2*OBS_DIM] → reshape to [H_KV*D_E, 2*OBS_DIM] → 列优先
    # GEMM: M=H_KV*D_E, K=OBS_DIM (K投影读前128, V投影读后128)
    sz = H_KV * D_E * 2 * OBS_DIM
    Wkv_2d = ref.Wkv.reshape(H_KV * D_E, 2 * OBS_DIM)
    packed[offset:offset+sz] = col_major(Wkv_2d)
    offset += sz

    # Wq [H_Q, D_E, D_H1] → reshape to [H_Q*D_E, D_H1] → 列优先
    sz = H_Q * D_E * D_H1
    Wq_2d = ref.Wq.reshape(H_Q * D_E, D_H1)
    packed[offset:offset+sz] = col_major(Wq_2d)
    offset += sz

    # bcat [H_Q*D_E] — 1D向量不变
    packed[offset:offset+BCAT_SIZE] = ref.bcat
    offset += BCAT_SIZE

    # Wico [D_H, 3*D_IN] 列优先
    sz = D_H * 3 * D_IN
    packed[offset:offset+sz] = col_major(ref.Wico)
    offset += sz

    # bico [3*D_H]
    packed[offset:offset+BICO_SIZE] = ref.bico
    offset += BICO_SIZE

    # Wf [ACT_DIM, D_H2] 列优先
    sz = ACT_DIM * D_H2
    packed[offset:offset+sz] = col_major(ref.Wf)
    offset += sz

    # bf [ACT_DIM]
    packed[offset:offset+BF_SIZE] = ref.bf
    offset += BF_SIZE

    # Wc1 [D_C, OBS_DIM] 列优先
    sz = D_C * OBS_DIM
    packed[offset:offset+sz] = col_major(ref.Wc1)
    offset += sz

    # bc1 [D_C]
    packed[offset:offset+BC1_SIZE] = ref.bc1
    offset += BC1_SIZE

    # Wc2 [D_C, D_C+INNER_DIM] 列优先
    sz = D_C * (D_C + INNER_DIM)
    packed[offset:offset+sz] = col_major(ref.Wc2)
    offset += sz

    # bc2 [D_C]
    packed[offset:offset+BC2_SIZE] = ref.bc2
    offset += BC2_SIZE

    # Wc3 [D_C, D_C] 列优先
    sz = D_C * D_C
    packed[offset:offset+sz] = col_major(ref.Wc3)
    offset += sz

    # bc3 [D_C]
    packed[offset:offset+BC3_SIZE] = ref.bc3
    offset += BC3_SIZE

    # Wc4 [1, D_C] 列优先
    sz = 1 * D_C
    packed[offset:offset+sz] = col_major(ref.Wc4)
    offset += sz

    # bc4 [1]
    packed[offset:offset+BC4_SIZE] = ref.bc4
    offset += BC4_SIZE

    assert offset == WEIGHT_FLOATS, f"Offset {offset} != {WEIGHT_FLOATS}"
    return packed


def pack_grads_col_major(ref):
    """打包梯度（与权重相同布局）"""
    grad_ref = type('GradRef', (), {})()
    grad_ref.Wkv  = ref.grad_Wkv
    grad_ref.Wq   = ref.grad_Wq
    grad_ref.bcat = ref.grad_bcat
    grad_ref.Wico = ref.grad_Wico
    grad_ref.bico = ref.grad_bico
    grad_ref.Wf   = ref.grad_Wf
    grad_ref.bf   = ref.grad_bf
    grad_ref.Wc1  = ref.grad_Wc1
    grad_ref.bc1  = ref.grad_bc1
    grad_ref.Wc2  = ref.grad_Wc2
    grad_ref.bc2  = ref.grad_bc2
    grad_ref.Wc3  = ref.grad_Wc3
    grad_ref.bc3  = ref.grad_bc3
    grad_ref.Wc4  = ref.grad_Wc4
    grad_ref.bc4  = ref.grad_bc4
    return pack_weights_col_major(grad_ref)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'golden_attnlstm.bin'
    print(f"Generating AttnLSTM golden data → {out_path}")

    rng = np.random.RandomState(42)
    ref = AttnLSTMReference(rng)

    # 输入数据
    observe = rng.randn(OBS_DIM, OBS_N).astype(np.float32) * 0.5
    inner   = rng.randn(INNER_DIM).astype(np.float32) * 0.5
    h_prev  = np.zeros(D_H, dtype=np.float32)
    c_prev  = np.zeros(D_H, dtype=np.float32)

    # 前向
    act, value, h_new, c_new, cache = ref.forward(observe, inner, h_prev, c_prev)

    # 上游梯度
    g_act   = rng.randn(ACT_DIM).astype(np.float32) * 0.5
    g_value = rng.randn(1).astype(np.float32).item() * 0.5

    # 反向
    g_observe_actor, g_h_prev_actor, g_c_prev_actor = ref.backward_actor(g_act, cache)
    g_observe = ref.backward_critic(g_value, cache, g_observe_actor)

    # 打包
    weights_packed = pack_weights_col_major(ref)
    grads_packed   = pack_grads_col_major(ref)

    # ================================================================
    # 写二进制文件
    # ================================================================
    with open(out_path, 'wb') as f:
        # Magic + version
        f.write(b'ATLN')  # AttLstm Network
        f.write(struct.pack('i', 1))

        # 维度头
        f.write(struct.pack('iiiiiiiiiiiii',
            OBS_N, OBS_DIM, INNER_DIM, ACT_DIM, D_E, H_KV, H_Q,
            D_H, D_H1, D_H2, D_IN, D_C, WEIGHT_FLOATS))

        def write_arr(name, arr):
            data = np.asarray(arr, dtype=np.float32).flatten()
            f.write(struct.pack('i', data.size))
            f.write(data.tobytes())
            print(f"  wrote {name}: {data.size} floats ({data.nbytes} bytes)")

        write_arr('weights', weights_packed)
        write_arr('observe', observe)
        write_arr('inner', inner)
        write_arr('h_prev', h_prev)
        write_arr('c_prev', c_prev)
        write_arr('act_out', act)
        write_arr('value_out', np.array([value], dtype=np.float32))
        write_arr('h_new', h_new)
        write_arr('c_new', c_new)
        write_arr('g_act', g_act)
        write_arr('g_value', np.array([g_value], dtype=np.float32))
        write_arr('grads', grads_packed)
        write_arr('g_observe', g_observe)
        write_arr('g_h_prev', g_h_prev_actor)
        write_arr('g_c_prev', g_c_prev_actor)

    size = Path(out_path).stat().st_size
    print(f"\nTotal: {size} bytes")
    print(f"Act range: [{act.min():.4f}, {act.max():.4f}]")
    print(f"Value: {value:.4f}")
    print("Done.")


if __name__ == '__main__':
    main()
