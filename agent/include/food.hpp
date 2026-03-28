/**
 * @file food.hpp
 * @brief 食物 Food —— 动物死亡后产生，随时间衰败
 */

#pragma once

#include "creature.hpp"

class Food : public Creature {
protected:
    float m_energy     = 0.f;   // 当前能量
    float m_decay_rate = 0.f;   // 衰败速率

public:
    Food() = default;

    Food(float x, float y, float energy)
        : Creature(EntityType::Food, x, y,
                   AgentConfig::ENTITY_RADIUS_DEFAULT,
                   AgentConfig::FOOD_HEALTH_MAX,
                   AgentConfig::FOOD_MAX_AGE)
        , m_energy(energy)
        , m_decay_rate(AgentConfig::FOOD_DECAY_RATE)
    {}

    // --- 访问器 ---
    float getEnergy() const { return m_energy; }

    // --- 更新（衰败） ---
    void update(float dt) {
        if (!isAlive()) return;

        m_energy -= m_decay_rate * dt;
        if (m_energy <= 0.f) {
            m_energy = 0.f;
            kill();
        }

        updateAge(dt);
    }

    // --- 消耗：返回实际获取的能量（经效率折算） ---
    float consume(float amount) {
        float actual = (amount < m_energy) ? amount : m_energy;
        m_energy -= actual;
        if (m_energy <= 0.f) {
            m_energy = 0.f;
            kill();
        }
        return actual * AgentConfig::ETA_PRED;
    }
};
