/**
 * @file observe.hpp
 * @brief 环境观测数据结构
 */

#pragma once

#include "entity.hpp"
#include "config.hpp"
#include <cstring>

/// 单个被感知实体的信息
struct ObserveEntry {
    float rel_x    = 0.f;   // 相对自身的x偏移（归一化）
    float rel_y    = 0.f;   // 相对自身的y偏移（归一化）
    float distance = 0.f;   // 距离（归一化）
    float type[static_cast<int>(EntityType::COUNT)] = {};  // one-hot 类型编码

    /// 观测条目总维度
    static constexpr int DIM = 3 + static_cast<int>(EntityType::COUNT);

    void setType(EntityType t) {
        std::memset(type, 0, sizeof(type));
        type[static_cast<int>(t)] = 1.0f;
    }
};

/// 单个智能体的完整环境观测
struct Observation {
    static constexpr int MAX_ENTRIES = NNConfig::OBSERVE_ENTRY_N;

    ObserveEntry entries[MAX_ENTRIES] = {};   // (M, observe_entry_dim) 二维矩阵
    int valid_count = 0;                      // 实际有效的实体数

    float state[NNConfig::STATE_DIM]     = {};  // 自身状态: [health, energy, hunger, breed_value]
    float special[NNConfig::SPECIAL_DIM] = {};  // 环境状态: [time_of_day, weather, temperature, light_level]

    void clear() {
        std::memset(entries, 0, sizeof(entries));
        valid_count = 0;
        std::memset(state, 0, sizeof(state));
        std::memset(special, 0, sizeof(special));
    }
};
