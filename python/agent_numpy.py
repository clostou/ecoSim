import numpy as np

from agent_config import AgentConfig


class ActivateFunc:
    """
    人工神经元的激活函数
    """

    def fn(self, z):
        """
        计算激活值
        :param z: 输入值
        :return: 神经元的激活值
        """
        pass

    def d(self, a):
        """
        计算导数项
        :param a: 激活值
        :return: 激活值对输入值的导数
        """
        pass


class Identity(ActivateFunc):

    def fn(self, z):
        return z

    def d(self, a):
        return np.ones_like(a)


class Sigmoid(ActivateFunc):

    def fn(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def d(self, a):
        return np.multiply(a, 1.0 - a)


class Tanh(ActivateFunc):

    def fn(self, z):
        return np.tanh(z)

    def d(self, a):
        return 1.0 - a**2


class ReLU(ActivateFunc):

    def fn(self, z):
        z[z < 0] = 0.0
        return z

    def d(self, a):
        _a = np.zeros_like(a)
        _a[a > 0] = 1.0
        return _a


class NetworkLayer:
    """
    各类神经网络层
    """

    def forward(self, x):
        """
        前向计算
        :param x: 上一层的输出值，为按列排布的二维numpy数组
        :return: 该层的输出值
        """
        pass

    def backward(self, g):
        """
        反向传播
        :param g: 该层输出值的误差梯度，为按列排布的二维numpy数组
        :return: 该层输入值的误差梯度
        """
        pass


class Linear(NetworkLayer):

    def __init__(self, input_channel, output_channel, activate=Identity(), dtype=None):
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.activate = activate
        self.w = np.random.normal(loc=0.0, scale=1.0/np.sqrt(input_channel),
                                  size=(output_channel, input_channel)).astype(dtype)
        self.b = np.random.normal(loc=0.0, scale=1.0,
                                  size=(output_channel, 1)).astype(dtype)
        self.input = np.zeros((input_channel, 1), dtype=dtype)
        self.output = np.zeros((output_channel, 1), dtype=dtype)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)

    def forward(self, x):
        self.input = x
        self.output = self.activate.fn(np.dot(self.w, x) + self.b)
        return self.output

    def backward(self, g):
        g_1 = np.multiply(self.activate.d(self.output), g)  # dE/da
        g_2 = np.dot(self.w.T, g_1)
        self.grad_w += np.dot(g_1, self.input.T)
        self.grad_b += np.sum(g_1, axis=1, keepdims=True)
        return g_2


class GQA:
    """
    分组查询注意力（Grouped Query Attention），若不给定`num_kv_pair`的值，则默认为多头注意力。

    注1：输出为原始查询的拼接结果，不包含投影；
    注2：采用了原始的反向传播实现，对于长序列内存消耗较大，进一步改进可参考FlashAttention。
    """

    def __init__(self, query_dim, key_dim, embed_dim, num_query, num_kv_pair=None, dtype=None):
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        if num_kv_pair is None:
            num_kv_pair = num_query
        query_per_head = max(num_query // num_kv_pair, 1)
        self.head_kv = num_kv_pair
        self.head_q = query_per_head
        self.output_dim = query_per_head * num_kv_pair * embed_dim

        # 注意力权重
        self.w_q = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(query_dim),
                                    size=(query_per_head, num_kv_pair, embed_dim, query_dim)).astype(dtype)
        self.w_k = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(key_dim),
                                    size=(num_kv_pair, embed_dim, key_dim)).astype(dtype)
        self.w_v = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(key_dim),
                                    size=(num_kv_pair, embed_dim, key_dim)).astype(dtype)
        # 注意力偏置（最后一个维度为预留的batch维度）
        self.b_q = np.random.normal(loc=0.0, scale=1.0,
                                    size=(query_per_head, num_kv_pair, embed_dim, 1, 1)).astype(dtype)
        self.b_k = np.random.normal(loc=0.0, scale=1.0,
                                    size=(num_kv_pair, embed_dim, 1, 1)).astype(dtype)
        self.b_v = np.random.normal(loc=0.0, scale=1.0,
                                    size=(num_kv_pair, embed_dim, 1, 1)).astype(dtype)

        # 输入变量缓存
        self.x_q = np.zeros((query_dim, 1, 1), dtype=dtype)
        self.x_k = np.zeros((key_dim, 1, 1), dtype=dtype)
        # 前向计算的中间变量
        self.tmp_q = np.zeros((query_per_head, num_kv_pair, embed_dim, 1, 1), dtype=dtype)
        self.tmp_k = np.zeros((num_kv_pair, embed_dim, 1, 1), dtype=dtype)
        self.tmp_v = np.zeros((num_kv_pair, embed_dim, 1, 1), dtype=dtype)
        self.tmp_s = np.zeros((query_per_head, num_kv_pair, 1, 1, 1), dtype=dtype)
        self.tmp_p = np.zeros((query_per_head, num_kv_pair, 1, 1, 1), dtype=dtype)

        # 参数的累积梯度
        self.grad_w_q = np.zeros_like(self.w_q)
        self.grad_w_k = np.zeros_like(self.w_k)
        self.grad_w_v = np.zeros_like(self.w_v)
        self.grad_b_q = np.zeros_like(self.b_q)
        self.grad_b_k = np.zeros_like(self.b_k)
        self.grad_b_v = np.zeros_like(self.b_v)

    def forward(self, x_q, x_k):
        """
        前向计算
        :param x_q: 查询序列，为(query_dim, query_len, batch)形状的三维数组
        :param x_k: 源序列（值序列），为(key_dim, key_len, batch)形状的三维数组
        :return: 原始查询结果，为(num_query * embed_dim, query_len, batch)形状的三维数组，
                 可通过后接线性变换层将查询结果投影至指定维度
        """
        self.x_q = x_q
        self.x_k = x_k
        # 计算Q, K, V矩阵
        q = np.dot(self.w_q, x_q.swapaxes(0, 1)) + self.b_q  # (Hq, Hk, E, Lq, N)
        k = np.dot(self.w_k, x_k.swapaxes(0, 1)) + self.b_k  # (Hk, E, Lk, N)
        v = np.dot(self.w_v, x_k.swapaxes(0, 1)) + self.b_v
        # 缩放点积注意力(SDPA)的数值稳定性实现
        attn_raw = np.einsum('qkein, kejn -> qkijn', q, k) / np.sqrt(self.embed_dim)  # S  # 哑标约定: L ↔ i,j
        attn_exp = np.exp(attn_raw - np.max(attn_raw, axis=-2, keepdims=True))
        score = attn_exp / np.sum(attn_exp, axis=-2, keepdims=True)  # P: (Hq, Hk, Lq, Lk, N)
        out = np.einsum('qkijn, kejn -> qkein', score, v)  # O: (Hq, Hk, E, Lq, N)
        # 更新中间变量
        self.tmp_q = q
        self.tmp_k = k
        self.tmp_v = v
        self.tmp_s = attn_raw
        self.tmp_p = score
        # 向量拼接并返回
        return out.reshape((-1, ) + out.shape[-2:])

    def backward(self, g):
        """
        反向传播
        :param g: 查询结果的误差梯度，为(num_query * embed_dim, query_len, batch)形状的三维数组
        :return: 查询序列(query_dim, query_len, batch)和值序列(key_dim, query_len, batch)的误差梯度
        """
        # 计算Q、K、V的误差梯度
        g = g.reshape((self.head_q, self.head_kv, self.embed_dim, *g.shape[-2:]))  # dO: (Hq, Hk, E, Lq, N)
        g_v = np.einsum('qkijn, qkein -> kejn', self.tmp_p, g)  # 无需按头平均: / self.head_q
        g_1 = np.einsum('qkein, kejn -> qkijn', g, self.tmp_v)  # dP: (Hq, Hk, Lq, Lk, N)
        dp_ds = - np.einsum('qkian, qkibn -> qkiabn', self.tmp_p, self.tmp_p)
        ind = np.arange(self.tmp_p.shape[3])
        dp_ds[..., ind, ind, :] += self.tmp_p
        g_2 = np.einsum('qkian, qkiabn -> qkibn', g_1, dp_ds) / np.sqrt(self.embed_dim)  # dS * √E
        # dp_ds = np.multiply(self.tmp_p, 1.0 - self.tmp_p)
        # g_2 = np.multiply(g_1, dp_ds) / np.sqrt(self.embed_dim)
        g_k = np.einsum('qkein, qkijn -> kejn', self.tmp_q, g_2)
        g_q = np.einsum('qkijn, kejn -> qkein', g_2, self.tmp_k)
        # self._dp = g_1
        # self._ds = g_2 * np.sqrt(self.embed_dim)
        # 计算参数的梯度
        self.grad_w_v += np.einsum('kejn, sjn -> kes', g_v, self.x_k)  # 哑标约定: query_dim,key_dim ↔ r,s
        self.grad_b_v[..., 0, 0] += np.einsum('kejn -> ke', g_v)  # 无需按序列和批量平均: / (self.x_k.shape[1] * self.x_k.shape[2])
        self.grad_w_k += np.einsum('kejn, sjn -> kes', g_k, self.x_k)
        self.grad_b_k[..., 0, 0] += np.einsum('kejn -> ke', g_k)
        self.grad_w_q += np.einsum('qkein, rin -> qker', g_q, self.x_q)
        self.grad_b_q[..., 0, 0] += np.einsum('qkein -> qke', g_q)
        # 计算输入值的误差梯度
        g_x_k = np.einsum('kes, kejn -> sjn', self.w_k, g_k) + np.einsum('kes, kejn -> sjn', self.w_v, g_v)
        g_x_q = np.einsum('qker, qkein -> rin', self.w_q, g_q)
        return g_x_q, g_x_k


class LSTM:
    """
    长短期记忆网络（耦合遗忘门和输入门）
    """

    def __init__(self, input_size, hidden_size, dtype=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()

        # 门输入权重
        self.w_i = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(input_size),
                                    size=(hidden_size, input_size)).astype(dtype)
        self.w_c = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(input_size),
                                    size=(hidden_size, input_size)).astype(dtype)
        self.w_o = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(input_size),
                                    size=(hidden_size, input_size)).astype(dtype)
        # 门隐状态权重
        self.u_i = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(hidden_size),
                                    size=(hidden_size, hidden_size)).astype(dtype)
        self.u_c = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(hidden_size),
                                    size=(hidden_size, hidden_size)).astype(dtype)
        self.u_o = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(hidden_size),
                                    size=(hidden_size, hidden_size)).astype(dtype)
        # 门偏置
        self.b_i = np.random.normal(loc=0.0, scale=1.0,
                                    size=(hidden_size, 1)).astype(dtype)
        self.b_c = np.random.normal(loc=0.0, scale=1.0,
                                    size=(hidden_size, 1)).astype(dtype)
        self.b_o = np.random.normal(loc=0.0, scale=1.0,
                                    size=(hidden_size, 1)).astype(dtype)

        # 初始隐状态和细胞状态
        self.h = np.zeros((hidden_size, 1), dtype=dtype)
        self.c = np.zeros((hidden_size, 1), dtype=dtype)

        # 输入变量与内部状态缓存
        self.seq_n = 0
        self.x_list = []
        self.c_list = []
        self.h_list = []
        # 前向计算的中间变量
        self.tmp_o_list = []
        self.tmp_tanhc_list = []
        self.tmp_altc_list = []
        self.tmp_i_list = []

        # 参数的累积梯度
        self.grad_w_i = np.zeros_like(self.w_i)
        self.grad_w_c = np.zeros_like(self.w_c)
        self.grad_w_o = np.zeros_like(self.w_o)
        self.grad_u_i = np.zeros_like(self.u_i)
        self.grad_u_c = np.zeros_like(self.u_c)
        self.grad_u_o = np.zeros_like(self.u_o)
        self.grad_b_i = np.zeros_like(self.b_i)
        self.grad_b_c = np.zeros_like(self.b_c)
        self.grad_b_o = np.zeros_like(self.b_o)

    def _clear_tmp(self):
        self.tmp_o_list.clear()
        self.tmp_tanhc_list.clear()
        self.tmp_altc_list.clear()
        self.tmp_i_list.clear()

    def forward(self, x):
        """
        前向计算
        :param x: 输入序列，为(input_size, batch)形状的二维数组
        :return: 输出序列（隐状态），为(hidden_size, batch)形状的二维数组，
                 可通过后接线性变换层将输出序列投影至指定维度
        """
        self.seq_n = np.shape(x)[-1]
        self.x_list = np.split(x, self.seq_n, axis=-1)
        self.c_list = [self.c]
        self.h_list = [self.h]
        self._clear_tmp()
        # 隐状态循环
        for x in self.x_list:
            gate_i = self.sigmoid.fn(np.dot(self.w_i, x) + np.dot(self.u_i, self.h) + self.b_i)
            gate_o = self.sigmoid.fn(np.dot(self.w_o, x) + np.dot(self.u_o, self.h) + self.b_o)
            c_alt = self.tanh.fn(np.dot(self.w_c, x) + np.dot(self.u_c, self.h) + self.b_c)
            self.c = np.multiply(1.0 - gate_i, self.c) + np.multiply(gate_i, c_alt)
            tanh_c = self.tanh.fn(self.c)
            self.h = np.multiply(gate_o, tanh_c)
            self.c_list.append(self.c)
            self.h_list.append(self.h)
            # 储存中间变量
            self.tmp_i_list.append(gate_i)
            self.tmp_o_list.append(gate_o)
            self.tmp_altc_list.append(c_alt)
            self.tmp_tanhc_list.append(tanh_c)
        return np.concatenate(self.h_list[1:], axis=1)

    def backward(self, g):
        """
        反向传播
        :param g: 输出序列的误差梯度，为(hidden_size, batch)形状的二维数组
        :return: 输入序列的误差梯度，为(input_size, batch)形状的二维数组
        """
        assert np.shape(g) == (len(self.h), self.seq_n), \
            "Input gradient mismatch: expect %s but got %s" % ((len(self.h), self.seq_n), np.shape(g))
        g_h_t = np.zeros_like(self.h)
        g_c_t = np.zeros_like(self.c)
        g_x = []
        for g_h, \
            h_t_last, c_t_last, x_t, \
            tanhc_t, altc_t, o_t, i_t in reversed(list(zip(
                np.split(g, self.seq_n, axis=-1),
                self.h_list[: -1], self.c_list[: -1], self.x_list,
                self.tmp_tanhc_list, self.tmp_altc_list, self.tmp_o_list, self.tmp_i_list))):
            # 计算隐状态梯度
            g_h_t = g_h + g_h_t
            g_c_t = g_h_t * o_t * (1.0 - tanhc_t**2) + g_c_t
            # 计算中间梯度
            g_o = g_h_t * tanhc_t * o_t * (1.0 - o_t)
            g_altc = g_c_t * i_t * (1.0 - altc_t**2)
            g_i = g_c_t * (altc_t - c_t_last) * i_t * (1.0 - i_t)
            # 计算参数的梯度
            self.grad_w_o += np.dot(g_o, x_t.T)
            self.grad_u_o += np.dot(g_o, h_t_last.T)
            self.grad_b_o += g_o
            self.grad_w_c += np.dot(g_altc, x_t.T)
            self.grad_u_c += np.dot(g_altc, h_t_last.T)
            self.grad_b_c += g_altc
            self.grad_w_i += np.dot(g_i, x_t.T)
            self.grad_u_i += np.dot(g_i, h_t_last.T)
            self.grad_b_i += g_i
            # 计算上一步的隐状态梯度（时间传递）
            g_h_t = np.dot(self.u_o.T, g_o) + np.dot(self.u_c.T, g_altc) + np.dot(self.u_i.T, g_i)
            g_c_t = g_c_t * (1.0 - i_t)
            # 计算输入值的误差梯度
            g_x.append(
                np.dot(self.w_o.T, g_o) + np.dot(self.w_c.T, g_altc) + np.dot(self.w_i.T, g_i)
            )
        self.seq_n = 0
        return np.concatenate(list(reversed(g_x)), axis=1)


class AttnLSTM(NetworkLayer):
    """
    门控注意力长短期记忆网络，架构与 PyTorch 版 AttnLSTM 完全一致：

    1. state_w: Linear(state_dim → hidden_state_dim) + tanh — 状态特征投影
    2. GQA 注意力: 从 h0 计算 Q，从 observe 计算 K/V，经 SDPA 提取观测特征
    3. LSTM 单步: 拼接 [h0, _state, _observe] 送入单步 LSTM（耦合输入/遗忘门）

    持久状态 h, c 在每次 forward 后更新，用于下一时间步。
    """

    def __init__(self, conf: AgentConfig):
        self.input_state_dim = conf.state_dim                       # 4
        self.input_observe_dim = conf.observe_entry_dim + conf.observe_entry_type_n  # 7
        self.hidden_dim = conf.hidden_dim                           # 32
        self.hidden_state_dim = conf.hidden_state_dim               # 16
        self.attn_dim = conf.attn_dim                               # 8
        self.attn_k = conf.attn_k                                   # 2
        self.attn_q = conf.attn_q                                   # 2
        self.attn_output_dim = conf.attn_dim * conf.attn_k * conf.attn_q  # 32
        lstm_input_dim = conf.hidden_dim + conf.hidden_state_dim + self.attn_output_dim  # 80

        # ---- 状态投影层 ----
        self.state_w = Linear(input_channel=conf.state_dim,
                              output_channel=conf.hidden_state_dim,
                              activate=Tanh(), dtype=conf.dtype)

        # ---- 注意力权重（遵循 GQA 的约定：embed_dim 在中间，输入特征维在最后） ----
        # K/V 权重: (Hk, E, D_obs)
        self.w_k = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(self.input_observe_dim),
                                    size=(conf.attn_k, conf.attn_dim, self.input_observe_dim)).astype(conf.dtype)
        self.w_v = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(self.input_observe_dim),
                                    size=(conf.attn_k, conf.attn_dim, self.input_observe_dim)).astype(conf.dtype)
        # Q 权重: (Hq, Hk, E, D_h)
        self.w_q = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(conf.hidden_dim),
                                    size=(conf.attn_q, conf.attn_k, conf.attn_dim, conf.hidden_dim)).astype(conf.dtype)
        # K/V 偏置: (Hk, E, 1) — 末维为序列长度方向（广播用）
        self.b_k = np.random.normal(loc=0.0, scale=1.0,
                                    size=(conf.attn_k, conf.attn_dim, 1)).astype(conf.dtype)
        self.b_v = np.random.normal(loc=0.0, scale=1.0,
                                    size=(conf.attn_k, conf.attn_dim, 1)).astype(conf.dtype)
        # Q 偏置: (Hq, Hk, E, 1)
        self.b_q = np.random.normal(loc=0.0, scale=1.0,
                                    size=(conf.attn_q, conf.attn_k, conf.attn_dim, 1)).astype(conf.dtype)

        # ---- LSTM 权重（耦合输入/遗忘门，与 agent_numpy.LSTM 一致） ----
        self.w_i = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(lstm_input_dim),
                                    size=(conf.hidden_dim, lstm_input_dim)).astype(conf.dtype)
        self.w_c = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(lstm_input_dim),
                                    size=(conf.hidden_dim, lstm_input_dim)).astype(conf.dtype)
        self.w_o = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(lstm_input_dim),
                                    size=(conf.hidden_dim, lstm_input_dim)).astype(conf.dtype)
        self.u_i = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(conf.hidden_dim),
                                    size=(conf.hidden_dim, conf.hidden_dim)).astype(conf.dtype)
        self.u_c = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(conf.hidden_dim),
                                    size=(conf.hidden_dim, conf.hidden_dim)).astype(conf.dtype)
        self.u_o = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(conf.hidden_dim),
                                    size=(conf.hidden_dim, conf.hidden_dim)).astype(conf.dtype)
        self.b_i = np.random.normal(loc=0.0, scale=1.0,
                                    size=(conf.hidden_dim, 1)).astype(conf.dtype)
        self.b_c = np.random.normal(loc=0.0, scale=1.0,
                                    size=(conf.hidden_dim, 1)).astype(conf.dtype)
        self.b_o = np.random.normal(loc=0.0, scale=1.0,
                                    size=(conf.hidden_dim, 1)).astype(conf.dtype)

        # ---- 持久隐状态 ----
        self.h = np.zeros((conf.hidden_dim, 1), dtype=conf.dtype)
        self.c = np.zeros((conf.hidden_dim, 1), dtype=conf.dtype)

        # ---- 中间变量缓存（用于反向传播） ----
        self._tmp_observe: np.ndarray | None = None
        self._tmp_x: np.ndarray | None = None          # 拼接后的 LSTM 输入
        self._tmp_h_prev: np.ndarray | None = None
        self._tmp_c_prev: np.ndarray | None = None
        self._tmp_q: np.ndarray | None = None          # (Hq, Hk, E, 1)
        self._tmp_k: np.ndarray | None = None          # (Hk, E, N)
        self._tmp_v: np.ndarray | None = None          # (Hk, E, N)
        self._tmp_s: np.ndarray | None = None          # (Hq, Hk, 1, N) 注意力原始得分
        self._tmp_p: np.ndarray | None = None          # (Hq, Hk, 1, N) softmax 后
        self._tmp_gate_i: np.ndarray | None = None
        self._tmp_gate_o: np.ndarray | None = None
        self._tmp_c_alt: np.ndarray | None = None
        self._tmp_tanhc: np.ndarray | None = None

        # ---- 梯度累积器 ----
        # Attention
        self.grad_w_k = np.zeros_like(self.w_k)
        self.grad_w_v = np.zeros_like(self.w_v)
        self.grad_w_q = np.zeros_like(self.w_q)
        self.grad_b_k = np.zeros_like(self.b_k)
        self.grad_b_v = np.zeros_like(self.b_v)
        self.grad_b_q = np.zeros_like(self.b_q)
        # LSTM
        self.grad_w_i = np.zeros_like(self.w_i)
        self.grad_w_c = np.zeros_like(self.w_c)
        self.grad_w_o = np.zeros_like(self.w_o)
        self.grad_u_i = np.zeros_like(self.u_i)
        self.grad_u_c = np.zeros_like(self.u_c)
        self.grad_u_o = np.zeros_like(self.u_o)
        self.grad_b_i = np.zeros_like(self.b_i)
        self.grad_b_c = np.zeros_like(self.b_c)
        self.grad_b_o = np.zeros_like(self.b_o)

        # 激活函数实例
        self.sigmoid = Sigmoid()
        self.tanh_fn = Tanh()

    def forward(self, observe, state):
        """
        前向计算（单时间步）。

        :param observe: 观测矩阵，形状 (observe_dim, observe_n)
                        其中 observe_dim = observe_entry_dim + observe_entry_type_n
                        observe_n 为当前观测到的实体数（≤ observe_entry_n）
        :param state:   状态向量，形状 (state_dim, 1)
        :return: 输出隐状态，形状 (hidden_dim, 1)
        """
        h_prev = self.h
        c_prev = self.c

        # ---- 1. 状态投影: Linear(state_dim → hidden_state_dim) + tanh ----
        _state = self.state_w.forward(state)  # (hidden_state_dim, 1)

        # ---- 2. GQA 注意力 ----
        # K, V 从 observe 投影: (Hk, E, D_obs) @ (D_obs, N) + (Hk, E, 1) → (Hk, E, N)
        k = np.matmul(self.w_k, observe) + self.b_k
        v = np.matmul(self.w_v, observe) + self.b_v
        # Q 从隐状态投影: (Hq, Hk, E, D_h) @ (D_h, 1) + (Hq, Hk, E, 1) → (Hq, Hk, E, 1)
        q = np.matmul(self.w_q, h_prev) + self.b_q

        # 缩放点积注意力(SDPA)
        # s[q,k,1,j] = sum_e q[q,k,e,1] * k[k,e,j] / sqrt(E)
        s = np.einsum('qkei,kej->qkij', q, k) / np.sqrt(self.attn_dim)  # (Hq, Hk, 1, N)
        s_max = np.max(s, axis=-1, keepdims=True)
        p = np.exp(s - s_max) / np.sum(np.exp(s - s_max), axis=-1, keepdims=True)  # softmax
        # o[q,k,e,1] = sum_j p[q,k,1,j] * v[k,e,j]
        o = np.einsum('qkij,kej->qkei', p, v)  # (Hq, Hk, E, 1)
        _observe = o.reshape(-1, 1)  # (attn_output_dim, 1)

        # ---- 3. 拼接 LSTM 输入 ----
        x = np.concatenate([h_prev, _state, _observe], axis=0)  # (lstm_input_dim, 1)

        # ---- 4. 单步 LSTM（耦合输入/遗忘门） ----
        gate_i = self.sigmoid.fn(np.dot(self.w_i, x) + np.dot(self.u_i, h_prev) + self.b_i)
        gate_o = self.sigmoid.fn(np.dot(self.w_o, x) + np.dot(self.u_o, h_prev) + self.b_o)
        c_alt = self.tanh_fn.fn(np.dot(self.w_c, x) + np.dot(self.u_c, h_prev) + self.b_c)
        c_new = (1.0 - gate_i) * c_prev + gate_i * c_alt
        tanhc = self.tanh_fn.fn(c_new)
        h_new = gate_o * tanhc

        # ---- 5. 更新持久状态 ----
        self.h = h_new
        self.c = c_new

        # ---- 6. 缓存中间变量 ----
        self._tmp_observe = observe
        self._tmp_x = x
        self._tmp_h_prev = h_prev
        self._tmp_c_prev = c_prev
        self._tmp_q = q
        self._tmp_k = k
        self._tmp_v = v
        self._tmp_s = s
        self._tmp_p = p
        self._tmp_gate_i = gate_i
        self._tmp_gate_o = gate_o
        self._tmp_c_alt = c_alt
        self._tmp_tanhc = tanhc

        return h_new

    def backward(self, g):
        """
        反向传播（单时间步），使用缓存的前向中间变量计算所有参数的梯度。

        :param g: 输出隐状态的误差梯度，形状 (hidden_dim, 1)
        :return: (g_observe, g_state)
                 g_observe — 观测矩阵的误差梯度，形状 (observe_dim, observe_n)
                 g_state   — 状态向量的误差梯度，形状 (state_dim, 1)
        """
        observe = self._tmp_observe
        x = self._tmp_x
        h_prev = self._tmp_h_prev
        c_prev = self._tmp_c_prev
        q = self._tmp_q      # (Hq, Hk, E, 1)
        k = self._tmp_k      # (Hk, E, N)
        v = self._tmp_v      # (Hk, E, N)
        p = self._tmp_p      # (Hq, Hk, 1, N)
        gate_i = self._tmp_gate_i
        gate_o = self._tmp_gate_o
        c_alt = self._tmp_c_alt
        tanhc = self._tmp_tanhc

        # =====================================================================
        # 1. LSTM 单步反向
        # =====================================================================
        g_h_new = g

        # 通过 h_new = gate_o * tanh(c_new)
        g_c_new = g_h_new * gate_o * self.tanh_fn.d(tanhc)
        g_gate_o = g_h_new * tanhc * self.sigmoid.d(gate_o)

        # 通过 c_new = (1 - gate_i) * c_prev + gate_i * c_alt
        g_c_alt = g_c_new * gate_i * self.tanh_fn.d(c_alt)
        g_gate_i = g_c_new * (c_alt - c_prev) * self.sigmoid.d(gate_i)

        # LSTM 权重梯度累积
        self.grad_w_o += np.dot(g_gate_o, x.T)
        self.grad_u_o += np.dot(g_gate_o, h_prev.T)
        self.grad_b_o += g_gate_o
        self.grad_w_c += np.dot(g_c_alt, x.T)
        self.grad_u_c += np.dot(g_c_alt, h_prev.T)
        self.grad_b_c += g_c_alt
        self.grad_w_i += np.dot(g_gate_i, x.T)
        self.grad_u_i += np.dot(g_gate_i, h_prev.T)
        self.grad_b_i += g_gate_i

        # x 的梯度: 由 W_* @ x 反向传播
        g_x = (np.dot(self.w_o.T, g_gate_o) +
               np.dot(self.w_c.T, g_c_alt) +
               np.dot(self.w_i.T, g_gate_i))

        # h_prev 的梯度（来自递归连接 U_* @ h_prev）
        g_h_prev_U = (np.dot(self.u_o.T, g_gate_o) +
                      np.dot(self.u_c.T, g_c_alt) +
                      np.dot(self.u_i.T, g_gate_i))

        # 拆分 g_x: 分别对应 [h_prev | _state | _observe]
        g_h_prev_x = g_x[:self.hidden_dim]
        g_state_ = g_x[self.hidden_dim:self.hidden_dim + self.hidden_state_dim]
        g_observe_flat = g_x[self.hidden_dim + self.hidden_state_dim:]

        # =====================================================================
        # 2. 注意力反向
        # =====================================================================
        g_o = g_observe_flat.reshape(self.attn_q, self.attn_k, self.attn_dim, 1)  # (Hq, Hk, E, 1)

        # dL/dV: 通过 o = sum_j p * v
        g_v = np.einsum('qkij,qkei->kej', p, g_o)  # (Hk, E, N)
        # dL/dP: 通过 o = sum_e g_o * v  (注意此处 v 作为"权重"，g_o 作为"上游梯度")
        g_p = np.einsum('qkei,kej->qkij', g_o, v)  # (Hq, Hk, 1, N)

        # softmax 反向: g_s = p * (g_p - sum_j g_p_j * p_j) / sqrt(E)
        g_s = p * (g_p - np.sum(g_p * p, axis=-1, keepdims=True)) / np.sqrt(self.attn_dim)

        # dL/dK, dL/dQ: 通过 s = sum_e q * k / sqrt(E)
        g_k = np.einsum('qkei,qkij->kej', q, g_s)  # (Hk, E, N)
        g_q = np.einsum('qkij,kej->qkei', g_s, k)   # (Hq, Hk, E, 1)

        # -- 注意力参数梯度 --
        # V 参数
        self.grad_w_v += np.matmul(g_v, observe.T)        # (Hk, E, N) @ (N, D) → (Hk, E, D)
        self.grad_b_v += np.sum(g_v, axis=-1, keepdims=True)  # (Hk, E, 1)
        # K 参数
        self.grad_w_k += np.matmul(g_k, observe.T)
        self.grad_b_k += np.sum(g_k, axis=-1, keepdims=True)
        # Q 参数
        # w_q 梯度: (Hq, Hk, E, D_h)，通过 q = w_q @ h_prev 反向
        self.grad_w_q += np.einsum('qkei,di->qked', g_q, h_prev)
        self.grad_b_q += g_q  # (Hq, Hk, E, 1) 直接累加（序列维度为 1）

        # -- 注意力输入梯度 --
        # observe 梯度: 通过 K 和 V 的线性投影反向
        g_observe_atten = (np.einsum('ked,kej->dj', self.w_k, g_k) +
                           np.einsum('ked,kej->dj', self.w_v, g_v))  # (D_obs, N)
        # h_prev 梯度（来自注意力 Q 投影）
        g_h_prev_Q = np.einsum('qked,qkei->di', self.w_q, g_q)  # (D_h, 1)

        # =====================================================================
        # 3. 状态投影层反向
        # =====================================================================
        g_state = self.state_w.backward(g_state_)

        # h_prev 的总梯度（内部使用，用于 BPTT；此处仅计算不返回）
        # g_h_prev = g_h_prev_x + g_h_prev_U + g_h_prev_Q

        return g_observe_atten, g_state


class Network:

    def __init__(self, *layers: NetworkLayer):
        pass


if __name__ == '__main__':
    pass


