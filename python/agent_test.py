import numpy as np
import torch

from torch import nn
import agent_torch as at
import agent_numpy as an


def linear_validation(dtype=float, seed=None):
    print("=" * 16,
          "Test for Single Layer Forward Network",
          "=" * 16)
    in_dim = 10
    out_dim = 4
    in_n = 50
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
        y_1 = linear_1.forward(x_1)
        x_2 = torch.tensor(x_1.T, requires_grad=True)
        y_2 = linear_2(x_2)
        print("  output equals:",
              np.allclose(y_1, y_2.detach().numpy().T))
        # 反向传播验证
        grad_o_1 = np.random.rand(*y_1.shape).astype(dtype)
        grad_x_1 = linear_1.backward(grad_o_1)
        grad_w_1 = linear_1.grad_w
        grad_b_1 = linear_1.grad_b
        grad_o_2 = torch.tensor(grad_o_1.T)
        torch.multiply(y_2, grad_o_2).sum().backward()
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
          "Test for Multi Query Attention Network",
          "=" * 16)
    in_dim = 10
    query_len = 2
    key_len = 20
    head_n = 4
    embed_dim = 8
    in_n = 50
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
    y_1 = attn_1.forward(x_q_1, x_k_1)
    x_q_2 = torch.tensor(x_q_1.T, requires_grad=True)
    x_k_2 = torch.tensor(x_k_1.T, requires_grad=True)
    y_2 = attn_2(x_q_2, x_k_2)
    print("  output equals:",
          np.allclose(y_1, y_2.detach().numpy().T))
    # 反向传播验证
    grad_o_1 = np.random.rand(*y_1.shape).astype(dtype)
    grad_x_q_1, grad_x_k_1 = attn_1.backward(grad_o_1)
    grad_wq_1 = attn_1.grad_w_q
    grad_wk_1 = attn_1.grad_w_k
    grad_wv_1 = attn_1.grad_w_v
    grad_bq_1 = attn_1.grad_b_q
    grad_bk_1 = attn_1.grad_b_k
    grad_bv_1 = attn_1.grad_b_v
    grad_o_2 = torch.tensor(grad_o_1.T)
    torch.multiply(y_2, grad_o_2).sum().backward()
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


if __name__ == '__main__':
    # linear_validation()
    attention_validation()


