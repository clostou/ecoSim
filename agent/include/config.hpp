/**
 * @file config.hpp
 * @brief 全局配置常量
 */

#pragma once

#include <cstdint>

namespace AgentConfig {
    // --- 实体基础属性 ---
    constexpr float ENTITY_RADIUS_DEFAULT = 5.0f;

    // --- 植物参数 ---
    constexpr float PLANT_HEALTH_MAX       = 100.0f;
    constexpr float PLANT_GROWTH_RATE      = 0.5f;       // 每秒基础生长速率
    constexpr float PLANT_SPREAD_COOLDOWN  = 30.0f;      // 扩张冷却（秒）
    constexpr float PLANT_WIND_SPREAD_PROB = 0.001f;     // 风传播概率（每步每空格）
    constexpr float PLANT_MAX_DENSITY      = 1.0f;       // 最大密度（影响视线遮挡）
    constexpr float PLANT_MAX_AGE          = 600.0f;     // 植物最大寿命（秒）

    // --- 猎物参数 ---
    constexpr float PREY_HEALTH_MAX        = 100.0f;
    constexpr float PREY_ENERGY_MAX        = 50.0f;
    constexpr float PREY_SPEED_MAX         = 3.0f;
    constexpr float PREY_VISION_RANGE      = 50.0f;
    constexpr float PREY_VISION_ANGLE      = 2.5f;       // 弧度（约143度）
    constexpr float PREY_ATK               = 20.0f;
    constexpr float PREY_MAX_AGE           = 300.0f;
    constexpr float PREY_BREED_THRESHOLD   = 80.0f;

    // --- 捕食者参数 ---
    constexpr float PRED_HEALTH_MAX        = 120.0f;
    constexpr float PRED_ENERGY_MAX        = 80.0f;
    constexpr float PRED_SPEED_MAX         = 3.5f;
    constexpr float PRED_VISION_RANGE      = 100.0f;
    constexpr float PRED_VISION_ANGLE      = 3.0f;       // 弧度（约172度）
    constexpr float PRED_ATK               = 85.0f;
    constexpr float PRED_MAX_AGE           = 400.0f;
    constexpr float PRED_BREED_THRESHOLD   = 100.0f;

    // --- 食物参数 ---
    constexpr float FOOD_DECAY_RATE        = 0.1f;       // 每秒衰减率
    constexpr float FOOD_HEALTH_MAX        = 50.0f;
    constexpr float FOOD_MAX_AGE           = 120.0f;

    // --- 能量传递效率 ---
    constexpr float ETA_PLANT = 0.01f;     // 光合作用效率
    constexpr float ETA_PREY  = 0.1f;      // 猎物进食效率
    constexpr float ETA_FOOD  = 0.8f;      // 死亡转化效率
    constexpr float ETA_PRED  = 0.5f;      // 捕食者进食效率

    // --- 能量消耗参数 ---
    constexpr float BASE_METABOLISM   = 0.5f;    // 基础代谢消耗/秒
    constexpr float MOVEMENT_COST     = 0.1f;    // 运动消耗系数
    constexpr float AGING_COST        = 0.2f;    // 年龄衰减消耗
    constexpr float HUNGER_RATE       = 0.3f;    // 饥饿度增长/秒
    constexpr float HUNGER_THRESHOLD  = 80.0f;   // 饥饿伤害阈值
    constexpr float STARVATION_DAMAGE = 5.0f;    // 饥饿伤害/秒

    // --- 繁殖参数 ---
    constexpr float BREED_ENERGY_COST  = 30.0f;  // 繁殖能量消耗
    constexpr float BREED_HEALTH_MIN   = 30.0f;  // 繁殖最低生命值
    constexpr float BREED_MIN_AGE      = 30.0f;  // 繁殖最小年龄
    constexpr float BREED_MAX_AGE_FRAC = 0.8f;   // 繁殖最大年龄（最大寿命的比例）
    constexpr float BREED_VALUE_RATE   = 0.2f;   // 繁育值增长/秒
}

namespace NNConfig {
    constexpr int OBSERVE_ENTRY_N       = 10;    // 可感知最大实体数 M
    constexpr int OBSERVE_ENTRY_DIM     = 3;     // 实体特征维度（不含类型）
    constexpr int OBSERVE_ENTRY_TYPE_N  = 5;     // 实体类型数量（one-hot）
    constexpr int STATE_DIM             = 4;     // 状态特征维度
    constexpr int SPECIAL_DIM           = 4;     // 环境状态维度
    constexpr int ACTION_DIM            = 2;     // 动作空间维度
}
