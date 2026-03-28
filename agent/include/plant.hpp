/**
 * @file plant.hpp
 * @brief 植物 Plant —— 生产者，随时间生长、扩张
 */

#pragma once

#include "creature.hpp"

class Plant : public Creature {
protected:
    float m_growth          = 0.f;    // 当前生长量 [0, 1]
    float m_growth_rate     = 0.f;    // 基础生长速率
    float m_density         = 0.f;    // 当前密度 [0, 1]（影响视线遮挡）
    float m_spread_timer    = 0.f;    // 扩张计时器
    float m_spread_cooldown = 0.f;    // 扩张冷却时间

public:
    Plant() = default;

    Plant(float x, float y)
        : Creature(EntityType::Plant, x, y,
                   AgentConfig::ENTITY_RADIUS_DEFAULT,
                   AgentConfig::PLANT_HEALTH_MAX,
                   AgentConfig::PLANT_MAX_AGE)
        , m_growth(0.f)
        , m_growth_rate(AgentConfig::PLANT_GROWTH_RATE)
        , m_density(0.f)
        , m_spread_timer(0.f)
        , m_spread_cooldown(AgentConfig::PLANT_SPREAD_COOLDOWN)
    {}

    // --- 访问器 ---
    float getGrowth()  const { return m_growth; }
    float getDensity() const { return m_density; }

    bool canSpread() const {
        return m_growth >= 1.0f && m_spread_timer >= m_spread_cooldown;
    }

    void resetSpreadTimer() { m_spread_timer = 0.f; }

    // --- 更新 ---
    void update(float dt, float light_level, float growth_modifier) {
        if (!isAlive()) return;

        // 生长受光照和气候调制
        float effective_rate = m_growth_rate * light_level * growth_modifier;
        m_growth += effective_rate * dt;
        if (m_growth > 1.0f) m_growth = 1.0f;

        // 密度与生长同步
        m_density = m_growth * AgentConfig::PLANT_MAX_DENSITY;

        // 半径随生长变化
        m_radius = AgentConfig::ENTITY_RADIUS_DEFAULT * (0.5f + 0.5f * m_growth);

        // 扩张计时
        m_spread_timer += dt;

        // 年龄更新
        updateAge(dt);
    }

    // --- 被食用：返回实际转化后的能量 ---
    float beEaten(float amount) {
        float actual = (amount < m_health) ? amount : m_health;
        m_health -= actual;
        m_growth -= actual / m_max_health;
        if (m_growth < 0.f) m_growth = 0.f;
        if (m_health <= 0.f) {
            m_health = 0.f;
            kill();
        }
        return actual * AgentConfig::ETA_PREY;
    }
};
