import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from torchinfo import summary

from agent_config import AgentConfig


class MQA(nn.Module):
    """多查询注意力"""

    def __init__(self, query_n, embed_dim, key_dim, query_dim, device=None, dtype=None):
        super(MQA, self).__init__()
        self.head_q = query_n
        self.embed_dim = embed_dim
        key_n = 1  # 由于pytorch提供的SDPA不支持多头注意力，该值固定为1

        # 注意力权重 (H, D, E)
        self.attn_wk = nn.Parameter(torch.rand(key_n, key_dim, embed_dim, device=device, dtype=dtype))
        self.attn_wv = nn.Parameter(torch.rand(key_n, key_dim, embed_dim, device=device, dtype=dtype))
        self.attn_wq = nn.Parameter(torch.rand(key_n * query_n, query_dim, embed_dim, device=device, dtype=dtype))
        self.attn_bk = nn.Parameter(torch.rand(key_n, 1, embed_dim, device=device, dtype=dtype))
        self.attn_bv = nn.Parameter(torch.rand(key_n, 1, embed_dim, device=device, dtype=dtype))
        self.attn_bq = nn.Parameter(torch.rand(key_n * query_n, 1, embed_dim, device=device, dtype=dtype))

    def forward(self, x_q, x_k):
        # 输入矩阵 (N, L, D) -> 注意力矩阵 (N, H, L, E)
        k = torch.stack([x_k_i @ self.attn_wk + self.attn_bk for x_k_i in x_k], dim=0)
        v = torch.stack([x_k_i @ self.attn_wv + self.attn_bv for x_k_i in x_k], dim=0)
        q = torch.stack([x_q_i @ self.attn_wq + self.attn_bq for x_q_i in x_q], dim=0)
        # 注意力输出 (N, H, L, E), 假定k=1
        out = F.scaled_dot_product_attention(q, k, v)
        # attn_raw = torch.einsum('nqie, nkje -> nqij', q, k) / math.sqrt(self.embed_dim)  # S
        # attn_exp = torch.exp(attn_raw - torch.max(attn_raw, dim=-1, keepdim=True)[0])
        # score = attn_exp / torch.sum(attn_exp, dim=-1, keepdim=True)  # P: (N, Hq, Lq, Lk)
        # out = torch.einsum('nqij, nkje -> nqie', score, v)  # O
        # self._s, self._p = attn_raw, score
        # self._s.retain_grad()
        # self._p.retain_grad()
        return torch.cat([out[:, i] for i in range(self.head_q)], dim=-1)  # (N, L, E * H)


class AttnLSTM(nn.Module):
    
    def __init__(self, conf: AgentConfig):
        super(AttnLSTM, self).__init__()
        self.input_state_dim = conf.state_dim
        self.input_observe_dim = conf.observe_entry_dim + conf.observe_entry_type_n
        attn_output_dim = conf.attn_dim * conf.attn_k * conf.attn_q

        self.state_w = nn.Linear(in_features=conf.state_dim, out_features=conf.hidden_state_dim,
                                 device=conf.device, dtype=conf.dtype)

        # 注意力权重 (D, E, H)
        self.attn_wk = nn.Parameter(torch.rand(conf.attn_k, self.input_observe_dim, conf.attn_dim,
                                               device=conf.device, dtype=conf.dtype))
        self.attn_wv = nn.Parameter(torch.rand(conf.attn_k, self.input_observe_dim, conf.attn_dim,
                                               device=conf.device, dtype=conf.dtype))
        self.attn_wq = nn.Parameter(torch.rand(conf.attn_k * conf.attn_q, conf.hidden_dim, conf.attn_dim,
                                               device=conf.device, dtype=conf.dtype))
        self.attn_bk = nn.Parameter(torch.rand(conf.attn_k, 1, conf.attn_dim,
                                               device=conf.device, dtype=conf.dtype))
        self.attn_bv = nn.Parameter(torch.rand(conf.attn_k, 1, conf.attn_dim,
                                               device=conf.device, dtype=conf.dtype))
        self.attn_bq = nn.Parameter(torch.rand(conf.attn_k * conf.attn_q, 1, conf.attn_dim,
                                               device=conf.device, dtype=conf.dtype))

        self.lstm = nn.LSTM(input_size=conf.hidden_dim+conf.hidden_state_dim+attn_output_dim,
                            hidden_size=conf.hidden_dim, num_layers=1, batch_first=True,
                            device=conf.device, dtype=conf.dtype)

        self.h0 = torch.zeros(1, conf.hidden_dim,
                              device=conf.device, dtype=conf.dtype)
        self.c0 = torch.zeros(1, conf.hidden_dim,
                              device=conf.device, dtype=conf.dtype)

    def forward(self, observe: torch.Tensor, state: torch.Tensor):
        _state = F.tanh(self.state_w(state))
        # 注意力输入矩阵 (H, L, E)
        k = observe @ self.attn_wk + self.attn_bk
        v = observe @ self.attn_wv + self.attn_bv
        q = self.h0 @ self.attn_wq + self.attn_bq
        # 注意力输出 (H, L, E), L=1
        _observe = F.scaled_dot_product_attention(q, k, v).flatten()
        # LSTM计算
        x = torch.cat([self.h0.squeeze(0), _state, _observe]).unsqueeze(0)
        y, (self.h0, self.c0) = self.lstm(x, (self.h0, self.c0))
        return y


class Actor(nn.Module):

    def __init__(self, conf: AgentConfig):
        super(Actor, self).__init__()
        self.action_variance = torch.ones(conf.action_dim,
                                          device=conf.device,
                                          dtype=conf.dtype) * conf.action_variance
        self.output_dim = conf.action_dim

        self.base_net = AttnLSTM(conf)
        self.linear_out = nn.Linear(in_features=conf.hidden_dim, out_features=conf.action_dim,
                                    device=conf.device, dtype=conf.dtype)

    def forward(self, observe: torch.Tensor, state: torch.Tensor):
        h = self.base_net(observe, state)
        y = F.sigmoid(self.linear_out(h))
        dist = Normal(y, self.action_variance)
        action = dist.sample()
        return action, dist.log_prob(action)


class Critic(nn.Module):

    def __init__(self, conf: AgentConfig):
        super(Critic, self).__init__()
        # 状态层权重
        self.linear_state = nn.Linear(in_features=conf.state_dim,
                                      out_features=conf.hidden_state_dim,
                                      device=conf.device, dtype=conf.dtype)
        # 观测层权重
        self.linear_observe = nn.Linear(in_features=conf.observe_entry_dim+conf.observe_entry_type_n,
                                        out_features=conf.hidden_dim,
                                        device=conf.device, dtype=conf.dtype)
        # 隐藏层
        hidden_list = [
            nn.Linear(in_features=conf.hidden_dim + conf.hidden_state_dim,
                      out_features=conf.hidden_dim,
                      device=conf.device, dtype=conf.dtype),
            nn.ReLU()
        ]
        for _ in range(conf.layer_n - 1):
            hidden_list.extend([
                nn.Linear(in_features=conf.hidden_dim,
                          out_features=conf.hidden_dim,
                          device=conf.device, dtype=conf.dtype),
                nn.ReLU()
            ])
        self.linear_hidden = nn.Sequential(*hidden_list)
        # 输出层权重
        self.linear_out = nn.Linear(in_features=conf.hidden_dim, out_features=1,
                                    device=conf.device, dtype=conf.dtype)

    def forward(self, observe: torch.Tensor, state: torch.Tensor):
        _state = F.sigmoid(self.linear_state(state))
        _observe, _ = torch.max(F.sigmoid(self.linear_observe(observe)), dim=0)  # 最大池化
        x = torch.cat([_state, _observe])
        y = self.linear_out(self.linear_hidden(x))
        return y


if __name__ == '__main__':
    config = AgentConfig(
        attn_q=4,
        attn_k=1,  # scaled_dot_product_attention不支持k≠1（不支持多头注意力）
        dtype=torch.float32
    )
    actor = Actor(config)
    critic = Critic(config)

    summary(actor, device='cuda')
    summary(critic, device='cuda')

    state_i = torch.rand((4, ), device='cuda')
    observe_i = torch.rand((10, 7), device='cuda')
    a, p = actor(observe_i, state_i)
    v = critic(observe_i, state_i)


