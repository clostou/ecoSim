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
        self.output_dim = query_per_head * num_kv_pair

        # 注意力权重
        self.w_q = np.random.normal(loc=0.0, scale=1.0 / np.sqrt(query_dim),
                                    size=(query_per_head, num_kv_pair, embed_dim, query_dim)).astype(dtype)
        self.w_k = np.random.normal(loc=0.0, scale=1.0/np.sqrt(key_dim),
                                    size=(num_kv_pair, embed_dim, key_dim)).astype(dtype)
        self.w_v = np.random.normal(loc=0.0, scale=1.0/np.sqrt(key_dim),
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

    def __init__(self):
        pass

    def forward(self):
        pass
    
    def backward(self):
        pass


class AttnLSTM(NetworkLayer):

    def __init__(self, conf: AgentConfig):
        pass


class Network:

    def __init__(self, *layers: NetworkLayer):
        pass


if __name__ == '__main__':
    pass


