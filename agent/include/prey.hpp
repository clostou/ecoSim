/**
 * @file prey.hpp
 * @brief 猎物 Prey —— 第一级消费者，以植物为食
 */

#pragma once

#include "animal.hpp"

class Prey : public Animal {
public:
    Prey() = default;

    Prey(float x, float y)
        : Animal(EntityType::Prey, x, y,
                 AgentConfig::ENTITY_RADIUS_DEFAULT,
                 AgentConfig::PREY_HEALTH_MAX,
                 AgentConfig::PREY_MAX_AGE,
                 AgentConfig::PREY_SPEED_MAX,
                 AgentConfig::PREY_ENERGY_MAX,
                 AgentConfig::PREY_ATK,
                 AgentConfig::PREY_VISION_RANGE,
                 AgentConfig::PREY_VISION_ANGLE,
                 AgentConfig::PREY_BREED_THRESHOLD)
    {}

    /// 从 Plant 中进食
    void eatPlant(Plant& plant) {
        float gained = plant.beEaten(m_energy_max - m_energy);
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
