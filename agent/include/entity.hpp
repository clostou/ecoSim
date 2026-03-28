/**
 * @file entity.hpp
 * @brief 实体基类 Entity —— 所有环境实体的公共属性
 *
 * Entity 不持有自身 ID（由 civ::Vector 分配和管理），
 * 但持有类型标签、位置、半径和存活状态。
 */

#pragma once

#include <cstdint>
#include "utils.hpp"

// ---------------------------------------------------------------------------
// EntityType 枚举
// ---------------------------------------------------------------------------
enum class EntityType : uint8_t {
    Wall      = 0,   // 地理隔离
    Plant     = 1,   // 植物
    Food      = 2,   // 食物（动物死亡后产生）
    Predator  = 3,   // 捕食者
    Prey      = 4,   // 猎物
    COUNT     = 5    // 类型数量（用于 one-hot 编码）
};

// ---------------------------------------------------------------------------
// EntityRef — 跨类型实体引用（类型 + civ::ID）
// ---------------------------------------------------------------------------
struct EntityRef {
    EntityType type   = EntityType::Wall;
    uint64_t   id     = 0;
};

// ---------------------------------------------------------------------------
// Entity 基类
// ---------------------------------------------------------------------------
class Entity {
protected:
    EntityType m_type;
    float      m_x      = 0.f;
    float      m_y      = 0.f;
    float      m_radius = 0.f;
    bool       m_alive  = true;

public:
    Entity() : m_type(EntityType::Wall) {}

    Entity(EntityType type, float x, float y, float radius)
        : m_type(type), m_x(x), m_y(y), m_radius(radius), m_alive(true) {}

    // --- 访问器 ---
    EntityType getType()   const { return m_type; }
    Vec2f      getPos()    const { return {m_x, m_y}; }
    float      getX()      const { return m_x; }
    float      getY()      const { return m_y; }
    float      getRadius() const { return m_radius; }
    bool       isAlive()   const { return m_alive; }

    // --- 修改器 ---
    void setPos(float x, float y) { m_x = x; m_y = y; }
    void setPos(const Vec2f& p)   { m_x = p.x; m_y = p.y; }
    void kill()                   { m_alive = false; }
};
