/**
 * @file creature.hpp
 * @brief 有机生物 Creature —— 具有生命值和年龄的实体
 */

#pragma once

#include "entity.hpp"
#include "config.hpp"

class Creature : public Entity {
protected:
    float m_health     = 0.f;
    float m_max_health = 0.f;
    float m_age        = 0.f;
    float m_max_age    = 0.f;

public:
    Creature() = default;

    Creature(EntityType type, float x, float y, float radius,
             float max_health, float max_age)
        : Entity(type, x, y, radius)
        , m_health(max_health)
        , m_max_health(max_health)
        , m_age(0.f)
        , m_max_age(max_age)
    {}

    // --- 访问器 ---
    float getHealth()    const { return m_health; }
    float getMaxHealth() const { return m_max_health; }
    float getAge()       const { return m_age; }
    float getMaxAge()    const { return m_max_age; }

    // --- 受击 ---
    void beAttacked(float damage) {
        m_health -= damage;
        if (m_health <= 0.f) {
            m_health = 0.f;
            kill();
        }
    }

    // --- 存活判定（覆盖 Entity） ---
    bool isAlive() const {
        return m_alive && m_health > 0.f;
    }

    // --- 年龄更新 ---
    void updateAge(float dt) {
        m_age += dt;
        if (m_age >= m_max_age) {
            m_health = 0.f;
            kill();
        }
    }
};
