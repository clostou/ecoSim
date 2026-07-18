# AttnLSTM(Actor) 与 FCN(Critic) 的 Warp 级 GEMM 分解设计

> 项目：ecoSim — 生态仿真中的 RL Agent 在线学习
> 日期：2026-07-12
> 依据：`agent/src/net_config.cuh`、`agent/src/net_kernel.cuh`、`doc/fnn-warp-gemm-design.md`、`python/agent_numpy.py`、`python/agent_torch.py`

---

## 一、背景与执行模型

ecoSim 中每个 Agent 拥有一套独立的神经网络（AttnLSTM Actor + FCN Critic），需要在 GPU 上以"**单 block 单网络、大量 block 并行**"的方式批量推理与在线学习。延续 `doc/fnn-warp-gemm-design.md` 的成熟模式：

| 需求 | 实现 |
|------|------|
| 单网络在单 block 内完成 | block = 4 warps × 32 threads = 128 threads |
| 大量网络并行 | grid = `num_networks`，每 block ↔ 一个网络 |
| 计算前先加载至共享内存 | `AttnLstmWeights` + `AttnLstmCache` 双缓冲 global→smem |
| 两类并行 | **GEMM 并行**（warp 分 M 行）+ **注意力并行**（warp 分头） |
| 多次前向 → 一次反向 | 多个 `AttnLstmCache` 串行生成，反向逐个累加梯度 |
| 计算后写回 global | 权重/梯度/持久状态 (h, c) 回写 |
| 性能关注点 | coalesced 访问、bank conflict 消除 |

### 默认配置维度（`DefaultConfig`）

| 符号 | 值 | 含义 |
|------|----|------|
| `M` (OBS_N) | 16 | 观测实体数（序列长度） |
| `D_obs` (OBS_DIM) | 8 | 单实体观测特征维 |
| `D_inn` (INNER_DIM) | 8 | 内部状态维（state+env，送入 LSTM） |
| `d_o` (ACT_DIM) | 2 | 动作维 |
| `d_e` (ATTN_EMBED_DIM) | 8 | 注意力嵌入维 |
| `h_kv` (ATTN_KV_HEADS) | 2 | KV 头数 |
| `h_q` (ATTN_Q_HEADS) | 4 | Q 头总数 |
| `q/kv` (ATTN_QUERY_N) | 2 | 每 KV 头的查询数 = `h_q/h_kv` |
| `d_h` (LSTM_HIDDEN_DIM) | 16 | LSTM 隐/细胞状态维 |
| `d_h1` (LSTM_QUERY_DIM) | 12 | Q 投影输入维（`h_prev` 切片） |
| `d_h2` (LSTM_OUTPUT_DIM) | 8 | LSTM 输出维（`h_new` 切片） |
| `d_c` (CRITIC_HIDDEN_DIM) | 16 | Critic 隐藏维 |
| `LSTM_INPUT_DIM` | 56 | `h_q*d_e + D_inn + d_h` = 32+8+16 |
| `BLOCK_DIM` / `WARPS` | 128 / 4 | |

---

## 二、与 Python 参考实现的对比

| 维度 | NumPy `AttnLSTM` | Torch `AttnLSTM` | **C++ 目标（以 C++ 为准）** |
|------|------------------|------------------|---------------------------|
| LSTM 门 | 耦合输入/遗忘门 `c=(1-i)c_prev + i*c_alt` | `nn.LSTM` 分离门 (i,f,g,o) | **沿用耦合门**（参考 NumPy 反向公式） |
| 注意力 | GQA 手写 SDPA，独立 b_q/b_k/b_v | `F.scaled_dot_product_attention`，仅 k=1（MQA） | GQA，`h_kv=2`，**K/V 无偏置**，`bcat` 仅输出偏置 |
| 状态投影 | `state_w: Linear(state→hidden_state_dim)+tanh` | 同 NumPy | **去除** state_w，`D_inn` 直接送入 LSTM 输入 |
| Q 投影输入 | 完整 `hidden_dim` | `h0`（完整） | **`h_prev[0:12]` 切片**（解耦） |
| LSTM 输出 | 完整 `h_new` | 完整 `h0` | **`h_new[8:16]` 切片 → Wf → action** |
| 递归权重 U | `U_*` 独立矩阵 | `nn.LSTM` 内部 | **U_* 折叠进 `Wico`**（`h_prev` 拼入 x） |
| 偏置 | q/k/v 各一组 | q/k/v 各一组 | `bcat[h_q*d_e]`：注意力输出偏置（post-reshape） |

> **设计要点**：C++ 通过"U 折叠 + 切片解耦 + 去投影层"压缩参数量与算子数，以适应单 block 内的共享内存预算。下文所有 GEMM 分解均以 C++ 布局为准。

---

## 三、数据布局回顾

直接依据 `agent/src/net_kernel.cuh`。**权重列优先**（与 `fnn_kernel.cuh` 的行优先不同，GEMM 索引需转置处理，见 §四）；**缓存行优先**；均 `alignas(16)`。

### 3.1 `AttnLstmWeights`（权重 + 梯度累加器，列优先）

| 字段 | 逻辑形状 | 打包轴 | 说明 |
|------|----------|--------|------|
| `Wkv` | `[h_kv, d_e, 2*D_obs]` | 末维：`[Wk \| Wv]` | K、V 权重沿输入维并列 |
| `Wq` | `[h_q, d_e, d_h1]` | — | Q 投影：`d_h1→d_e`，按 h_q 头 |
| `bcat` | `[h_q*d_e]` | — | 注意力**输出**偏置（加在 reshape 后的 `_observe`） |
| `Wico` | `[d_h, 3*d_in]` | 末维：`[Wi \| Wc \| Wo]` | 三门权重（`U_*` 已折叠进 `d_in` 的 `h_prev` 段） |
| `bico` | `[3*d_h]` | `[bi \| bc \| bo]` | 三门偏置 |
| `Wf` / `bf` | `[d_o, d_h2]` / `[d_o]` | — | Actor 输出层 |
| `Wc1`/`bc1` | `[d_c, D_obs]` / `[d_c]` | — | Critic L1 |
| `Wc2`/`bc2` | `[d_c, d_c+D_inn]` / `[d_c]` | — | Critic L2（输入 `[c_h1 \| inner]`） |
| `Wc3`/`bc3` | `[d_c, d_c]` / `[d_c]` | — | Critic L3 |
| `Wc4`/`bc4` | `[1, d_c]` / `[1]` | — | Critic 输出层 |

梯度字段同名前缀 `grad_`，结构与权重一致，在反向中跨多 cache 累加。

### 3.2 `AttnLstmCache`（激活缓存，行优先）

| 字段 | 形状 | 用途（前向写 / 反向读） |
|------|------|------------------------|
| `x_kv` | `[D_obs, M]` | 观测矩阵；反向算 `grad_Wk/Wv` |
| `q` | `[h_q, d_e]` | Q 投影输出；反向算 `grad_Wq`、`dK` |
| `k` / `v` | `[h_kv, d_e, M]` | K/V 投影；反向算 `grad_Wk/Wv`、`dQ`、`dObserve` |
| `p` | `[h_q, h_kv, M]` | softmax 概率；反向 softmax 反推 `dS` |
| `x` | `[d_in=56]` | 拼接 LSTM 输入；反向算 `grad_Wico` |
| `h_prev` / `c_prev` | `[d_h]` | 前一步状态；反向算 `grad_U`（折叠进 `Wico`）与 `dQ` |
| `gate_i` / `gate_o` | `[d_h]` | sigmoid 门激活；反向门导数 |
| `c_alt` / `tanhc` | `[d_h]` | tanh 激活；反向门导数 |
| `lstm_o` | `[d_h2=8]` | `h_new[8:16]` 切片；反向算 `grad_Wf` |
| `act` | `[d_o]` | Actor 输出 |
| `c_h1`/`c_h2`/`c_h3` | `[d_c]` | Critic 各层激活；反向算 `grad_Wc*` |

> 持久状态 `h`、`c`（16 维）存于 global memory 的 per-network 状态区，前向前加载 `h_prev`/`c_prev`，反向后写回 `h_new`/`c_new`。

---

## 四、基本 GEMM 算子清单

沿用 `test/test-cutlass/src/fnn_warp_gemm.cuh` 三件套，并扩展注意力专用算子。**关键差异**：本设计权重为列优先，故 GEMM 索引由 `W[m*K + k]`（行优先）改为 `W[k*M + m]`（列优先），lane 读取改为沿 K 维连续——仍 coalesced。

### 4.1 前向 GEMM（列优先版）
```
warp_gemm_forward(W_gmem_col, X_smem, C_smem, M, K, N)
  C[M,N] = W[M,K] @ X[K,N]   (W 列优先：W[k*M + m])
  - M 按 warp 分组（GEMM 并行）
  - 每 lane 取 K 的 stride-32 段，warp_reduce_sum 后 lane0 写 C
```

### 4.2 反向 GEMM
- `warp_gemm_backward_input`：`GX[K,N] = W^T @ GY`，线程粒度平铺 (K,N)，W 按列读（列优先下相邻线程 k 相邻 → coalesced）。
- `warp_gemm_backward_weight`：`GW[M,K] += GY @ X^T`，线程粒度平铺 (M,K)，每 (m,k) 单线程写无需原子。

### 4.3 新增：三门 fused GEMM（适配 `Wico[d_h, 3*d_in]`）
```
warp_gemm_3gate_forward(Wico_gmem, x_smem, gate_i, gate_c, gate_o, M=d_h, K=d_in)
  for m in warp 分配的行:
    acc_i = acc_c = acc_o = 0
    for k = lane; k < K; k += 32:           # x 只加载一次，复用 3×
      acc_i += Wico[m*3K + k]        * x[k]  # 列块 0
      acc_c += Wico[m*3K + K + k]    * x[k]  # 列块 1
      acc_o += Wico[m*3K + 2K + k]   * x[k]  # 列块 2
    acc_* = warp_reduce_sum(acc_*)
    if lane==0: gate_i[m]=σ(acc_i); gate_c[m]=tanh(acc_c); gate_o[m]=σ(acc_o)
```
> 注：`Wico` 列优先 `[d_h, 3*d_in]` 时行 m 的三段连续，正好支持上述单遍三累加；x 复用 3 次省 2/3 带宽。反向同理可写 `warp_gemm_3gate_backward_*`。

### 4.4 注意力并行算子（warp 分头）
每 warp 处理一对 `(h_q, h_kv)`（共 4 对，恰好 4 warp）：
- `attn_qk_gemm`：`s[1, M] = Q[1, d_e] @ K[d_e, M] / √d_e`（K 按 h_kv 共享）。
- `softmax_fwd`：数值稳定减最大、exp、归一化 → `p[1, M]`。
- `attn_pv_gemm`：`o[1, d_e] = P[1, M] @ V[M, d_e]`（V 按 h_kv 共享）。
- 反向：`softmax_bwd`（`dS = P*(dP - ΣdP·P)/√d_e`）、`dQ = dS @ K^T`、`dK = dS^T @ Q`、`dV = P^T @ dO`，参考 `agent_numpy.py:GQA.backward`。

---

## 五、AttnLSTM(Actor) 前向 GEMM 分解（单时间步）

| # | 算子 | 形状 (M,K,N, batch) | 输入 | 输出 → cache | 备注 |
|---|------|---------------------|------|--------------|------|
| 1 | `warp_gemm_forward` ×2 | (d_e=8, D_obs=8, M=16, h_kv=2) | `observe`(`x_kv`) | `k`, `v` | Wkv 拆 Wk\|Wv；observe 复用 |
| 2 | `warp_gemm_forward` | (h_q*d_e=32, d_h1=12, 1) | `h_prev[0:12]` | `q` | 无偏置 |
| 3 | `attn_qk_gemm` | (1, d_e=8, M=16, h_q=4) | `q`, `k` | `s` | K 按 h_kv 共享 |
| 4 | `softmax_fwd` | (1, M=16, h_q*h_kv=4) | `s` | `p` | 减最大 + exp + 归一 |
| 5 | `attn_pv_gemm` | (d_e=8, M=16, 1, h_q=4) | `p`, `v` | `o` | V 按 h_kv 共享 |
| 6 | 元素式 | — | `o`(reshape→32) | `_observe` | `+ bcat[32]` |
| 7 | concat | — | `h_prev(16) \| inner(8) \| _observe(32)` | `x[56]` | 顺序须与 Wico 列块一致 |
| 8 | `warp_gemm_3gate_forward` | (d_h=16, d_in=56, ×3) | `x` | `gate_i, c_alt, gate_o` | `+ bico`；σ/tanh |
| 8b | 元素式 | — | `gate_i, c_alt, c_prev` | `c_new` | `c_new=(1-i)c_prev + i*c_alt` |
| 8c | 元素式 | — | `c_new, gate_o` | `h_new` | `tanhc=tanh(c_new); h_new=gate_o*tanhc` |
| 9 | `warp_gemm_forward` | (d_o=2, d_h2=8, 1) | `h_new[8:16]`→`lstm_o` | `act` | `+ bf`；sigmoid |
| 10 | 回写 | — | `h_new, c_new` | global 持久状态 | 供下一步前向 |

---

## 六、AttnLSTM(Actor) 反向 GEMM 分解（单 cache，逐 cache 累加）

输入 `g_act[d_o]`。逆序执行，所有 `grad_*` 在 smem 中累加，多 cache 循环后统一写回。

| # | 算子 | 公式 | 依赖 cache 字段 |
|---|------|------|-----------------|
| R1 | `activation_bwd<Sigmoid>` | `d_lstm_o = g_act * σ'(act)` | `act` |
| R2 | `warp_gemm_backward_weight` | `grad_Wf += d_lstm_o @ lstm_o^T` | `lstm_o` |
| R2b | `accumulate_bias_grad` | `grad_bf += d_lstm_o` | — |
| R3 | `warp_gemm_backward_input` | `d_z_o = Wf^T @ d_lstm_o` → scatter 进 `d_h_new[8:16]` | — |
| R4 | 元素式门导数（耦合门） | `d_gate_o = d_h_new * tanhc * σ'(gate_o)`; `d_c_new = d_h_new * gate_o * tanh'(tanhc)`; `d_c_alt = d_c_new * gate_i * tanh'(c_alt)`; `d_gate_i = d_c_new * (c_alt - c_prev) * σ'(gate_i)`; `d_c_prev += d_c_new * (1 - gate_i)` | `gate_i, gate_o, c_alt, tanhc, c_prev` |
| R5 | `warp_gemm_3gate_backward_weight` | `grad_Wico += [d_gate_i \| d_gate_c \| d_gate_o] @ x^T`（折叠形式） | `x` |
| R5b | `accumulate_bias_grad` | `grad_bico += [d_gate_i \| d_gate_c \| d_gate_o]` | — |
| R6 | `warp_gemm_3gate_backward_input` | `d_x = Wi^T d_gate_i + Wc^T d_gate_c + Wo^T d_gate_o` | — |
| R7 | 拆分 `d_x` | `d_h_prev(16) \| d_inner(8) \| d_observe_flat(32)` | — |
| R8 | `activation_bwd`（无，bcat 为线性偏置） | `d_observe_flat` 经 reshape → `d_o_attn[h_q, h_kv, d_e]` | — |
| R9 | `attn_pv_bwd` | `d_v = p^T @ d_o`; `d_p = d_o @ v^T` | `p, v` |
| R10 | `softmax_bwd` | `d_s = p*(d_p - Σ_j d_p_j*p_j)/√d_e` | `p` |
| R11 | `attn_qk_bwd` | `d_k = q^T @ d_s`; `d_q = d_s @ k^T` | `q, k` |
| R12 | `warp_gemm_backward_weight` ×2 | `grad_Wk += d_k @ observe^T`; `grad_Wv += d_v @ observe^T` | `x_kv` |
| R13 | `warp_gemm_backward_input` ×2 | `d_observe = Wk^T @ d_k + Wv^T @ d_v` | — |
| R14 | `warp_gemm_backward_weight` | `grad_Wq += d_q @ h_prev[0:12]^T` | `h_prev` |
| R15 | `warp_gemm_backward_input` | `d_h_prev[0:12] += Wq^T @ d_q` | — |
| R16 | 汇总 `d_h_prev` | `d_h_prev = (来自 R3 scatter) + (来自 R6 拆分) + (来自 R15)` | 供 BPTT 上一 cache |

> 多 cache：外层循环 `for cache in caches[::-1]`，`d_h_prev` 在 cache 间传递（BPTT）；`grad_*` 全程累加，循环结束写回 global。

---

## 七、FCN(Critic) GEMM 分解

前向 4 层 + 1 次 max-pool，激活均为 sigmoid（依 `AttnLstmCache` 注释）。

| 层 | 前向算子 | (M, K, N) | 输入 | 输出 → cache |
|----|----------|-----------|------|--------------|
| L1 | `warp_gemm_forward` + `b1` + sigmoid | (d_c=16, D_obs=8, M=16) | `observe` | `c_h1` |
| L1b | cooperative `max_reduce` | 沿 N=M=16 归约 | `c_h1[16,16]` | `c_h1[16]`（max-pool） |
| L2 | `warp_gemm_forward` + `b2` + sigmoid | (16, d_c+D_inn=24, 1) | `[c_h1 \| inner]` | `c_h2` |
| L3 | `warp_gemm_forward` + `b3` + sigmoid | (16, 16, 1) | `c_h2` | `c_h3` |
| L4 | `warp_gemm_forward` + `b4` | (1, 16, 1) | `c_h3` | `v`（线性输出） |

反向对每层用 `warp_gemm_backward_input/weight` + `activation_apply_bwd<Sigmoid>`（L4 无激活），模式与 `fnn_kernel.cuh` 完全一致，不再赘述。max-pool 反向：梯度仅传给前向选中的 argmax 位置（需 cache argmax 索引或重算）。

---

## 八、共享内存与并行策略

### 8.1 双缓冲加载
- 前向 kernel 启动时：block 协作将 `AttnLstmWeights`（权重，~3KB）从 global 加载到 smem 的权重区；当前步 `AttnLstmCache` 按需在 smem 激活区构建；持久 `h_prev/c_prev` 加载到 smem。
- 采用 **buf_a / buf_b 双缓冲**（参考 `fnn_kernel.cuh:SmemLayout`）：一层 GEMM 写 buf_b 时，下一层输入可从另一缓冲读取；层间 `__syncthreads` + 指针交换。
- 反向 kernel：同样加载权重 + 逐个 `AttnLstmCache`；`grad_*` 占用与权重等大的 smem 区，多 cache 累加后写回。

### 8.2 两类并行
- **GEMM 并行**（Wico、Wc* 等 M 较大的层）：M 按 warp 分组，warp 内 lane 协作点积，`warp_reduce_sum`。
- **注意力并行**（SDPA）：warp ↔ `(h_q, h_kv)` 对，4 warp 各处理一对；K/V 在同 h_kv 组内通过 smem 共享。

### 8.3 coalesced
- 权重列优先：`warp_gemm_forward` 中 lane L 读 `W[k*M + m]`，沿 K 步进 32——对固定 m，相邻 lane 的 k 相邻，地址连续 → coalesced。
- 缓存行优先：激活值按行连续，批量读 coalesced。

### 8.4 bank conflict
- smem 列访问（`backward_input` 按 K 步进读 W）易触发 32-bank 冲突 → 行尾加 `SMEM_PAD=32`（参考 `fnn_config.cuh`）。
- `warp_gemm_3gate` 三累加器读 `Wico` 的三个列块，同 m 同 k 三地址跨 3*K 间距，间距非 32 倍数 → 无冲突。
- 双缓冲两区间隔 ≥ 32 float 以错开 bank。

---

## 九、内存与启动配置

| 项目 | 估算 |
|------|------|
| `AttnLstmWeights`（权重+梯度） | ~6 KB |
| 单步 `AttnLstmCache` | ~1.5 KB |
| smem 激活双缓冲 | ~512 B |
| 持久状态 h/c | 128 B |
| 单 block smem 合计 | < 16 KB（含双缓冲余量） |
| grid | `num_networks` |
| block | 128 (4 warps) |
| cache 数 | 前向步数 `T`（每步一个 `AttnLstmCache`，存 global） |

---

## 十、待修 Bug 与开放问题

1. **`net_config.cuh` 断言方向错误**：`ATTN_KV_HEADS % ATTN_Q_HEADS == 0`（`2 % 4 ≠ 0`）失败；`ATTN_QUERY_N = h_kv/h_q = 0`。应改为 `ATTN_Q_HEADS % ATTN_KV_HEADS == 0`、`ATTN_QUERY_N = h_q/h_kv = 2`。
2. **`LSTM_INPUT_DIM` 分段顺序**：本设计约定 `x = [h_prev(16) | inner(8) | _observe(32)]`，须与 `Wico` 的 `d_in` 列分块一致；实现时需在 `net_kernel.cuh` 注释固化。
3. **max-pool argmax**：Critic L1 反向需前向 argmax；当前 `AttnLstmCache` 未存索引，需补字段或反向重算。
4. **`Wkv` 打包方向**：现按输入维 `[h_kv, d_e, 2*D_obs]` 并列，K/V 不能单 GEMM 融合输出；若未来允许调整布局为输出维并列可合并为单 GEMM。

---

## 十一、任务清单

- [ ] 修复 `net_config.cuh` 头数断言与 `ATTN_QUERY_N` 公式（§10.1）
- [ ] 实现列优先版 `warp_gemm_forward / backward_input / backward_weight`（§4.1–4.2）
- [ ] 实现 `warp_gemm_3gate_forward / 3gate_backward_*`（§4.3）
- [ ] 实现注意力并行算子 `attn_qk_gemm / softmax_fwd / attn_pv_gemm` 及反向（§4.4）
- [ ] Actor 前向 kernel（§五）
- [ ] Actor 反向 kernel（含多 cache BPTT 累加）（§六）
- [ ] Critic 前向/反向 kernel（含 max-pool + argmax cache）（§七）
- [ ] 双缓冲 smem 装载/回写（§8.1）
- [ ] host 端管理器（参考 `fnn_host.cuh:FNNHandle`：alloc/init/zero_grad/forward/backward/apply_sgd）
- [ ] NumPy golden 生成脚本（参考 `fnn/tools/generate_golden.py`）
- [ ] 正确性测试（逐算子相对误差 < 1e-5）+ 多网络独立性 + 性能基准

---

## 十二、关键代码索引

| 功能 | 现有参考文件 | 行号概览 |
|------|-------------|----------|
| 数据布局（权威） | `agent/src/net_kernel.cuh` | 全文 |
| 编译期配置 | `agent/src/net_config.cuh` | 全文 |
| `warp_reduce_sum` / 三件套 GEMM | `test/test-cutlass/src/fnn_warp_gemm.cuh` | L33–L170 |
| 双缓冲 `SmemLayout` / kernel 流程 | `test/test-cutlass/src/fnn_kernel.cuh` | L78–L358 |
| `SMEM_PAD` / bank-conflict 处理 | `test/test-cutlass/src/fnn_config.cuh` | L44–L49 |
| 激活函数 fwd/bwd | `test/test-cutlass/src/fnn_activations.cuh` | 全文 |
| GQA/LSTM 反向数学参考 | `python/agent_numpy.py` | `GQA`/`LSTM`/`AttnLSTM` |
| 前馈网络设计思想 | `doc/fnn-warp-gemm-design.md` | 全文 |
