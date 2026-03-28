/**
 * @file predator.hpp
 * @brief 捕食者 Predator —— 第二级消费者，通过攻击猎物获取食物
 */

#pragma once

#include "animal.hpp"

class Predator : public Animal {
public:
    Predator() = default;

    Predator(float x, float y)
        : Animal(EntityType::Predator, x, y,
                 AgentConfig::ENTITY_RADIUS_DEFAULT,
                 AgentConfig::PRED_HEALTH_MAX,
                 AgentConfig::PRED_MAX_AGE,
                 AgentConfig::PRED_SPEED_MAX,
                 AgentConfig::PRED_ENERGY_MAX,
                 AgentConfig::PRED_ATK,
                 AgentConfig::PRED_VISION_RANGE,
                 AgentConfig::PRED_VISION_ANGLE,
                 AgentConfig::PRED_BREED_THRESHOLD)
    {}

    /// 对 Creature 执行攻击
    void attackTarget(Creature& target) const {
        attack(target);
    }

    /// 从 Food 中进食
    void eatFood(Food& food) {
        float gained = food.consume(m_energy_max - m_energy);
        eat(gained);
    }

    /// 每步更新（简化版，不含神经网络决策）
    void update(float dt) {
        if (!isAlive()) return;
        applyVelocity(dt);
        updateEnergy(dt);
        updateAge(dt);
    }
};
