/**
 * @file net_config.cuh
 * @brief cuda 智能体网络的编译期配置文件
 */

#pragma once

namespace agent_gpu {

template <
    int N_ = 1,
    int M_ = 16,
    int D_obs_ = 8,  // observe
    int D_inn = 8,  // state + env
    int d_o_ = 2,
    int d_e_ = 8,
    int h_kv_ = 2,
    int h_q_ = 4,
    int d_h_ = 16,
    int d_h1_ = 12,
    int d_h2_ = 8,
    int d_c_ = 16,
    int block_size_ = 128,
    int a2c_steps_ = 10
>
struct AttnLstmConfig {
    static constexpr int NET_N = N_;                    /// 网络批量大小
    static constexpr int OBS_N = M_;                    /// 观测数量
    static constexpr int OBS_DIM = D_obs_;              /// 观测空间维度
    static constexpr int INNER_DIM = D_inn;             /// 内部状态维度
    static constexpr int ACT_DIM = d_o_;                /// 动作维度
    static_assert(NET_N == 1, "Batch networks are not supported yet, please set NET_N=1");

    static constexpr int ATTN_EMBED_DIM = d_e_;         /// 注意力维度
    static constexpr int ATTN_KV_HEADS = h_kv_;         /// 注意力KV头数
    static constexpr int ATTN_Q_HEADS = h_q_;           /// 注意力Q头数
    static constexpr int ATTN_QUERY_N = h_q_ / h_kv_;   /// 每个kv对的查询数
    static_assert(ATTN_Q_HEADS % ATTN_KV_HEADS == 0, "ATTN_Q_HEADS must be a multiple of ATTN_KV_HEADS");

    static constexpr int LSTM_INPUT_DIM =               /// LSTM输入维度
        h_q_ * d_e_ + D_inn + d_h_;
    static constexpr int LSTM_HIDDEN_DIM = d_h_;        /// LSTM隐藏层维度
    static constexpr int LSTM_QUERY_DIM = d_h1_;        /// LSTM隐藏层（查询解耦）维度
    static constexpr int LSTM_OUTPUT_DIM = d_h2_;       /// LSTM隐藏层（输出解耦）维度

    static constexpr int CRITIC_HIDDEN_DIM = d_c_;      /// Critic隐藏层维度
    
    static constexpr int NET_SIZE =                     /// 网络参数总量（GQA + LSTM + Output + Critic）
        NET_N * (
            (2 * ATTN_KV_HEADS * OBS_DIM + ATTN_Q_HEADS * LSTM_QUERY_DIM + ATTN_Q_HEADS) * ATTN_EMBED_DIM + 
            3 * (LSTM_INPUT_DIM + 1) * LSTM_HIDDEN_DIM + 
            (LSTM_OUTPUT_DIM + 1) * ACT_DIM + 
            ((OBS_DIM + 1) + (CRITIC_HIDDEN_DIM + INNER_DIM + 1) + (CRITIC_HIDDEN_DIM + 1) + 1) * CRITIC_HIDDEN_DIM + 1
        );
    static constexpr int TOTAL_BYTES =                  /// 网络缓存总量（GQA 输入 + GQA 激活 + LSTM 输入/激活 + Output 输入/激活 + Critic 激活）
        NET_N * (
            OBS_DIM * OBS_N + 
            (ATTN_Q_HEADS + 2 * ATTN_KV_HEADS * OBS_N) * ATTN_EMBED_DIM + ATTN_Q_HEADS * ATTN_KV_HEADS * OBS_N + 
            LSTM_INPUT_DIM + 2 * LSTM_HIDDEN_DIM + 4 * LSTM_HIDDEN_DIM + 
            LSTM_OUTPUT_DIM + ACT_DIM + 
            3 * CRITIC_HIDDEN_DIM
        );
    static constexpr int BLOCK_DIM   = block_size_;
    static constexpr int WARPS       = block_size_ / 32;
    static constexpr int SMEM_PAD    = 32;              /// 共享内存行尾填充以避免bank conflict
    static constexpr int A2C_STEPS   = a2c_steps_;      /// A2C 步长（梯度回传窗口）
    static_assert(BLOCK_DIM % 32 == 0, "BLOCK_DIM must be a multiple of 32");
};

using DefaultConfig = AttnLstmConfig<1, 16, 8, 8, 2, 8, 2, 4, 16, 12, 8, 16, 128, 10>;

}
