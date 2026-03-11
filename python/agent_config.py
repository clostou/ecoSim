__all__ = ['AgentConfig']


class AgentConfig:

    device = 'cuda'
    dtype = None

    # 基本设置
    observe_entry_n = 10  # 可感知到的最大实体数
    observe_entry_dim = 3  # 实体特征维度（不包含实体类型）
    observe_entry_type_n = 4  # 实体类型数量
    state_dim = 4  # 状态特征维度
    action_dim = 2  # 动作空间维度

    # 高级设置
    attn_dim = 8  # 注意力维度
    attn_k = 2  # 用于提取观测特征的键-值对数（注意力头的组数）
    attn_q = 2  # 用于提取观测特征的查询数（每组头的多查询数）
    hidden_state_dim = 16  # 升维后的状态特征维度
    hidden_dim = 32  # 隐状态的维度
    layer_n = 1  # LSTM块的堆叠数量以及评论员网络的层数
    action_variance = 0.05  # 动作函数的方差（固定值）

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


