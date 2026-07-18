import time

import numpy as np
import torch

from torch import nn
import agent_torch as at
import agent_numpy as an


class Profiler:

    def __init__(self, func_name: str, *data: [np.ndarray, torch.Tensor]):
        self._name = func_name
        self._bytes = 0
        for d in data:
            self._bytes += d.dtype.itemsize * np.prod(d.shape).item()
        self._time = 0.

    @staticmethod
    def _sec2str(second: float) -> str:
        ns = round(second * 1e9)  # 转换为纳秒整数
        if ns == 0:
            return "0s 0ms"

        # 单位表：(纳秒数, 符号)
        units = [
            (86400 * 10 ** 9, 'd'), (3600 * 10 ** 9, 'h'), (60 * 10 ** 9, 'm'),
            (10 ** 9, 's'), (10 ** 6, 'ms'), (10 ** 3, 'us'), (1, 'ns')
        ]

        # 找到第一个能容纳的单位（从大到小）
        first = len(units) - 1
        for i, (val, _) in enumerate(units):
            if ns >= val:
                first = i
                break

        val1, sym1 = units[first]
        num1 = ns // val1
        rem = ns % val1

        # 第二个单位取下一个（若没有只显示第一个单位）
        if first + 1 < len(units):
            val2, sym2 = units[first + 1]
            num2 = rem // val2
            return f"{num1}{sym1} {num2}{sym2}"
        else:
            return f"{num1}{sym1}"

    @staticmethod
    def _val2str(value: float) -> str:
        if value == 0:
            return "0.00"
        units = ["", "K", "M", "G", "T", "P"]
        sign = 1 if value >= 0 else -1
        v = abs(value)
        idx = 0
        # 当整数部分有3位或以上时进位（即 >=100）
        while v >= 100 and idx < len(units) - 1:
            v = round(v / 1000.0, 2)
            idx += 1
        v *= sign
        if idx == len(units) and abs(v) > 1000:
            return f"{v:,}{units[idx]}"
        else:
            return f"{v:.2f}{units[idx]}"

    def __enter__(self):
        self._time = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._time = time.perf_counter() - self._time
        if self._time == 0:
            print(f"{self._name}: {self._sec2str(self._time)}, --B/s")
        else:
            bw = self._bytes / self._time
            print(f"{self._name}: {self._sec2str(self._time)}, {self._val2str(bw)}B/s")


def linear_validation(dtype=float, seed=None):
    print("=" * 16,
          "Test for Single Layer Forward Network",
          "=" * 16)
    in_dim = 10
    out_dim = 4
    in_n = 10_000
    np.random.seed(seed)
    for i, (act_1, act_2) in enumerate(zip([an.Identity(), an.Sigmoid(), an.Tanh(), an.ReLU()],
                                           [nn.Identity(), nn.Sigmoid(), nn.Tanh(), nn.ReLU()]),
                                       start=1):
        print(f"{i}. {act_1.__class__.__name__}")
        # 创建网络
        linear_1 = an.Linear(input_channel=in_dim, output_channel=out_dim, activate=act_1, dtype=dtype)
        linear_2 = nn.Sequential(nn.Linear(in_features=in_dim, out_features=out_dim, dtype=dtype), act_2)
        # 初始参数复制
        with torch.no_grad():
            linear_2[0].weight.copy_(torch.tensor(linear_1.w))
            linear_2[0].bias.copy_(torch.tensor(linear_1.b.flatten()))
        # 前向计算验证
        x_1 = np.random.rand(in_dim, in_n).astype(dtype)
        with Profiler(f"[{linear_1.__class__.__name__}.forward]", x_1):
            y_1 = linear_1.forward(x_1)
        x_2 = torch.tensor(x_1.T, requires_grad=True)
        with Profiler(f"[{linear_2.__class__.__name__}.forward]", x_2):
            y_2 = linear_2(x_2)
        print("  output equals:",
              np.allclose(y_1, y_2.detach().numpy().T))
        # 反向传播验证
        grad_o_1 = np.random.rand(*y_1.shape).astype(dtype)
        with Profiler(f"[{linear_1.__class__.__name__}.backward]", grad_o_1):
            grad_x_1 = linear_1.backward(grad_o_1)
        grad_w_1 = linear_1.grad_w
        grad_b_1 = linear_1.grad_b
        grad_o_2 = torch.tensor(grad_o_1.T)
        target = torch.multiply(y_2, grad_o_2).sum()
        with Profiler(f"[{linear_2.__class__.__name__}.backward]", grad_o_2):
            target.backward()
        grad_x_2 = x_2.grad
        grad_w_2 = linear_2[0].weight.grad
        grad_b_2 = linear_2[0].bias.grad
        print("  input gradient equals:",
              np.allclose(grad_x_1, grad_x_2.numpy().T))
        print("  weight gradient equals:",
              np.allclose(grad_w_1, grad_w_2.numpy()))
        print("  bias gradient equals:",
              np.allclose(grad_b_1.flatten(), grad_b_2.numpy()))


def attention_validation(dtype=float, seed=None):
    print("=" * 16,
          "Test for Multi Query Attention (MQA) Network",
          "=" * 16)
    in_dim = 10
    query_len = 2
    key_len = 20
    head_n = 4
    embed_dim = 8
    in_n = 1_000
    np.random.seed(seed)
    # 创建网络
    attn_1 = an.GQA(query_dim=in_dim, key_dim=in_dim, embed_dim=embed_dim, num_query=head_n, num_kv_pair=1,
                    dtype=dtype)
    attn_2 = at.MQA(query_n=head_n, embed_dim=embed_dim, key_dim=in_dim, query_dim=in_dim,
                    dtype=dtype)
    # 初始参数复制
    with torch.no_grad():
        attn_2.attn_wk.copy_(torch.tensor(attn_1.w_k.swapaxes(1, 2)))
        attn_2.attn_bk.copy_(torch.tensor(attn_1.b_k[..., 0].swapaxes(1, 2)))
        attn_2.attn_wv.copy_(torch.tensor(attn_1.w_v.swapaxes(1, 2)))
        attn_2.attn_bv.copy_(torch.tensor(attn_1.b_v[..., 0].swapaxes(1, 2)))
        attn_2.attn_wq.copy_(torch.tensor(attn_1.w_q[:, 0].swapaxes(1, 2)))  # 默认num_kv_pair=1
        attn_2.attn_bq.copy_(torch.tensor(attn_1.b_q[:, 0][..., 0].swapaxes(1, 2)))  # 默认num_kv_pair=1
    # 前向计算验证
    x_q_1 = np.random.rand(in_dim, query_len, in_n).astype(dtype)
    x_k_1 = np.random.rand(in_dim, key_len, in_n).astype(dtype)
    with Profiler(f"[{attn_1.__class__.__name__}.forward]", x_q_1, x_k_1):
        y_1 = attn_1.forward(x_q_1, x_k_1)
    x_q_2 = torch.tensor(x_q_1.T, requires_grad=True)
    x_k_2 = torch.tensor(x_k_1.T, requires_grad=True)
    with Profiler(f"[{attn_2.__class__.__name__}.forward]", x_q_2, x_k_2):
        y_2 = attn_2(x_q_2, x_k_2)
    print("  output equals:",
          np.allclose(y_1, y_2.detach().numpy().T))
    # 反向传播验证
    grad_o_1 = np.random.rand(*y_1.shape).astype(dtype)
    with Profiler(f"[{attn_1.__class__.__name__}.backward]", grad_o_1):
        grad_x_q_1, grad_x_k_1 = attn_1.backward(grad_o_1)
    grad_wq_1 = attn_1.grad_w_q
    grad_wk_1 = attn_1.grad_w_k
    grad_wv_1 = attn_1.grad_w_v
    grad_bq_1 = attn_1.grad_b_q
    grad_bk_1 = attn_1.grad_b_k
    grad_bv_1 = attn_1.grad_b_v
    grad_o_2 = torch.tensor(grad_o_1.T)
    target = torch.multiply(y_2, grad_o_2).sum()
    with Profiler(f"[{attn_2.__class__.__name__}.backward]", grad_o_2):
        target.backward()
    grad_x_q_2, grad_x_k_2 = x_q_2.grad, x_k_2.grad
    grad_wq_2 = attn_2.attn_wq.grad
    grad_wk_2 = attn_2.attn_wk.grad
    grad_wv_2 = attn_2.attn_wv.grad
    grad_bq_2 = attn_2.attn_bq.grad
    grad_bk_2 = attn_2.attn_bk.grad
    grad_bv_2 = attn_2.attn_bv.grad
    # _grad_p_1 = attn_1._dp[:, 0]
    # _grad_s_1 = attn_1._ds[:, 0]
    # _grad_p_2 = attn_2._p.grad.numpy().transpose((1, 2, 3, 0))
    # _grad_s_2 = attn_2._s.grad.numpy().transpose((1, 2, 3, 0))
    print("  query gradient equals:",
          np.allclose(grad_x_q_1, grad_x_q_2.numpy().T))
    print("  key gradient equals:",
          np.allclose(grad_x_k_1, grad_x_k_2.numpy().T))
    print("  query weight gradient equals:",
          np.allclose(grad_wq_1, grad_wq_2.numpy().swapaxes(1, 2)[:, np.newaxis]))  # 默认num_kv_pair=1
    print("  key weight gradient equals:",
          np.allclose(grad_wk_1, grad_wk_2.numpy().swapaxes(1, 2)))
    print("  value weight gradient equals:",
          np.allclose(grad_wv_1, grad_wv_2.numpy().swapaxes(1, 2)))
    print("  query bias gradient equals:",
          np.allclose(grad_bq_1[..., 0], grad_bq_2.numpy().swapaxes(1, 2)[:, np.newaxis]))  # 默认num_kv_pair=1
    print("  key bias gradient equals:",
          np.allclose(grad_bk_1[..., 0], grad_bk_2.numpy().swapaxes(1, 2)))
    print("  value bias gradient equals:",
          np.allclose(grad_bv_1[..., 0], grad_bv_2.numpy().swapaxes(1, 2)))


def lstm_validation(dtype=float, seed=None):
    print("=" * 16,
          "Test for Long-short Term Memory (LSTM) Network",
          "=" * 16)
    in_dim = 16
    hidden_dim = 8
    in_n = 1_000
    np.random.seed(seed)
    # 创建网络
    lstm_1 = an.LSTM(input_size=in_dim, hidden_size=hidden_dim, dtype=dtype)
    lstm_2 = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, dtype=dtype)
    # 初始参数复制
    with torch.no_grad():
        lstm_2.weight_ih_l0.copy_(torch.tensor(np.concatenate([
            lstm_1.w_i, -lstm_1.w_i, lstm_1.w_c, lstm_1.w_o], axis=0)))  # 耦合输入门和遗忘门
        lstm_2.weight_hh_l0.copy_(torch.tensor(np.concatenate([
            lstm_1.u_i, -lstm_1.u_i, lstm_1.u_c, lstm_1.u_o], axis=0)))
        lstm_2.bias_ih_l0.copy_(torch.tensor(np.concatenate([
            lstm_1.b_i, -lstm_1.b_i, lstm_1.b_c, lstm_1.b_o], axis=0).flatten() / 2))  # 偏置一分为二
        lstm_2.bias_hh_l0.copy_(torch.tensor(np.concatenate([
            lstm_1.b_i, -lstm_1.b_i, lstm_1.b_c, lstm_1.b_o], axis=0).flatten() / 2))
    # 前向计算验证
    x_1 = np.random.rand(in_dim, in_n).astype(dtype)
    with Profiler(f"[{lstm_1.__class__.__name__}.forward]", x_1):
        y_1 = lstm_1.forward(x_1)
    hn_1, cn_1 = lstm_1.h, lstm_1.c
    x_2 = torch.tensor(x_1.T, requires_grad=True)
    with Profiler(f"[{lstm_2.__class__.__name__}.forward]", x_2):
        y_2, (hn_2, cn_2) = lstm_2(x_2)
    print("  output equals:",
          np.allclose(y_1, y_2.detach().numpy().T))
    print("  state equals:",
          np.allclose(hn_1, hn_2.detach().numpy().T) and \
          np.allclose(cn_1, cn_2.detach().numpy().T))
    # 反向传播验证
    grad_h_1 = np.random.rand(*y_1.shape).astype(dtype)
    with Profiler(f"[{lstm_1.__class__.__name__}.backward]", grad_h_1):
        grad_x_1 = lstm_1.backward(grad_h_1)
    grad_wi_1 = lstm_1.grad_w_i
    grad_wc_1 = lstm_1.grad_w_c
    grad_wo_1 = lstm_1.grad_w_o
    grad_ui_1 = lstm_1.grad_u_i
    grad_uc_1 = lstm_1.grad_u_c
    grad_uo_1 = lstm_1.grad_u_o
    grad_bi_1 = lstm_1.grad_b_i
    grad_bc_1 = lstm_1.grad_b_c
    grad_bo_1 = lstm_1.grad_b_o
    grad_h_2 = torch.tensor(grad_h_1.T)
    target = torch.multiply(y_2, grad_h_2).sum()
    with Profiler(f"[{lstm_2.__class__.__name__}.backward]", grad_h_2):
        target.backward()
    grad_x_2 = x_2.grad
    grad_wi_2 = lstm_2.weight_ih_l0.grad[: hidden_dim] - lstm_2.weight_ih_l0.grad[hidden_dim: 2 * hidden_dim]
    grad_wc_2 = lstm_2.weight_ih_l0.grad[2 * hidden_dim: 3 * hidden_dim]
    grad_wo_2 = lstm_2.weight_ih_l0.grad[3 * hidden_dim:]
    grad_ui_2 = lstm_2.weight_hh_l0.grad[: hidden_dim] - lstm_2.weight_hh_l0.grad[hidden_dim: 2 * hidden_dim]
    grad_uc_2 = lstm_2.weight_hh_l0.grad[2 * hidden_dim: 3 * hidden_dim]
    grad_uo_2 = lstm_2.weight_hh_l0.grad[3 * hidden_dim:]
    grad_bi_2 = lstm_2.bias_ih_l0.grad[: hidden_dim] - lstm_2.bias_ih_l0.grad[hidden_dim: 2 * hidden_dim]
    grad_bc_2 = lstm_2.bias_ih_l0.grad[2 * hidden_dim: 3 * hidden_dim]  # bias_ih_l0和bias_hh_l0的梯度相等
    grad_bo_2 = lstm_2.bias_ih_l0.grad[3 * hidden_dim:]
    print("  input gradient equals:",
          np.allclose(grad_x_1, grad_x_2.numpy().T))
    print("  gate_i weight gradient equals:",
          np.allclose(grad_wi_1, grad_wi_2.numpy()) and np.allclose(grad_ui_1, grad_ui_2.numpy()))
    print("  gate_i bias gradient equals:",
          np.allclose(grad_bi_1.flatten(), grad_bi_2.numpy()))
    print("  gate_c weight gradient equals:",
          np.allclose(grad_wc_1, grad_wc_2.numpy()) and np.allclose(grad_uc_1, grad_uc_2.numpy()))
    print("  gate_c bias gradient equals:",
          np.allclose(grad_bc_1.flatten(), grad_bc_2.numpy()))
    print("  gate_o weight gradient equals:",
          np.allclose(grad_wo_1, grad_wo_2.numpy()) and np.allclose(grad_uo_1, grad_uo_2.numpy()))
    print("  gate_o bias gradient equals:",
          np.allclose(grad_bo_1.flatten(), grad_bo_2.numpy()))


if __name__ == '__main__':
    linear_validation()
    attention_validation()
    lstm_validation()


