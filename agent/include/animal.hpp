/**
 * @file animal.hpp
 * @brief 动物 Animal —— 可移动的消费者，具有速度、能量、繁育能力
 */

#pragma once

#include "creature.hpp"
#include <array>
#include <algorithm>

class Animal : public Creature {
protected:
    float m_vel_x           = 0.f;
    float m_vel_y           = 0.f;
    float m_speed_max       = 0.f;
    float m_energy          = 0.f;
    float m_energy_max      = 0.f;
    float m_hunger          = 0.f;
    float m_breed_value     = 0.f;
    float m_breed_threshold = 0.f;
    float m_atk             = 0.f;
    float m_vision_range    = 0.f;
    float m_vision_angle    = 0.f;

public:
    Animal() = default;

    Animal(EntityType type, float x, float y, float radius,
           float max_health, float max_age,
           float speed_max, float energy_max,
           float atk, float vision_range, float vision_angle,
           float breed_threshold)
        : Creature(type, x, y, radius, max_health, max_age)
        , m_vel_x(0.f), m_vel_y(0.f)
        , m_speed_max(speed_max)
        , m_energy(energy_max)
        , m_energy_max(energy_max)
        , m_hunger(0.f)
        , m_breed_value(0.f)
        , m_breed_threshold(breed_threshold)
        , m_atk(atk)
        , m_vision_range(vision_range)
        , m_vision_angle(vision_angle)
    {}

    // --- 访问器 ---
    Vec2f getVelocity()     const { return {m_vel_x, m_vel_y}; }
    float getSpeedMax()     const { return m_speed_max; }
    float getEnergy()       const { return m_energy; }
    float getEnergyMax()    const { return m_energy_max; }
    float getHunger()       const { return m_hunger; }
    float getBreedValue()   const { return m_breed_value; }
    float getAtk()          const { return m_atk; }
    float getVisionRange()  const { return m_vision_range; }
    float getVisionAngle()  const { return m_vision_angle; }

    /// 返回状态向量 [health, energy, hunger, breed_value]
    std::array<float, 4> getState() const {
        return { m_health, m_energy, m_hunger, m_breed_value };
    }

    // --- 速度设置（由决策输出驱动） ---
    void setVelocity(float vx, float vy) {
        m_vel_x = vx;
        m_vel_y = vy;
        // 限速
        float speed_sq = vx * vx + vy * vy;
        if (speed_sq > m_speed_max * m_speed_max) {
            float scale = m_speed_max / std::sqrt(speed_sq);
            m_vel_x *= scale;
            m_vel_y *= scale;
        }
    }

    // --- 位置更新 ---
    void applyVelocity(float dt) {
        m_x += m_vel_x * dt;
        m_y += m_vel_y * dt;
    }

    // --- 攻击目标（对 Creature 造成 m_atk 伤害） ---
    void attack(Creature& target) const {
        target.beAttacked(m_atk);
    }

    // --- 进食（直接获取能量） ---
    void eat(float energy_gained) {
        m_energy += energy_gained;
        if (m_energy > m_energy_max) m_energy = m_energy_max;
        m_hunger -= energy_gained * 2.0f;
        if (m_hunger < 0.f) m_hunger = 0.f;
    }

    // --- 繁殖判定 ---
    bool canBreed() const {
        return m_breed_value >= m_breed_threshold
            && m_energy >= AgentConfig::BREED_ENERGY_COST
            && m_health > AgentConfig::BREED_HEALTH_MIN
            && m_age >= AgentConfig::BREED_MIN_AGE
            && m_age <= m_max_age * AgentConfig::BREED_MAX_AGE_FRAC;
    }

    // --- 繁殖后消耗 ---
    void onBreed() {
        m_energy      -= AgentConfig::BREED_ENERGY_COST;
        m_breed_value  = 0.f;
    }

    // --- 能量更新（每个时间步调用） ---
    void updateEnergy(float dt) {
        // 基础代谢消耗
        m_energy -= AgentConfig::BASE_METABOLISM * dt;

        // 运动消耗（与速度平方成正比）
        float speed_sq = m_vel_x * m_vel_x + m_vel_y * m_vel_y;
        m_energy -= AgentConfig::MOVEMENT_COST * speed_sq * dt;

        // 年龄衰减消耗（近似线性）
        float age_ratio = m_age / m_max_age;
        m_energy -= AgentConfig::AGING_COST * age_ratio * dt;

        // 饥饿度增加
        m_hunger += AgentConfig::HUNGER_RATE * dt;

        // 繁育值随时间积累
        m_breed_value += AgentConfig::BREED_VALUE_RATE * dt;

        // 饥饿导致生命值下降
        if (m_hunger > AgentConfig::HUNGER_THRESHOLD) {
            m_health -= AgentConfig::STARVATION_DAMAGE * dt;
        }

        // 能量不足时降速
        if (m_energy <= 0.f) {
            m_energy = 0.f;
            m_speed_max *= 0.5f;  // 惩罚系数
        }

        // 生命值归零则死亡
        if (m_health <= 0.f) {
            m_health = 0.f;
            kill();
        }
    }
};
