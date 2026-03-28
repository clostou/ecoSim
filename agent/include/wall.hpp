/**
 * @file wall.hpp
 * @brief 地理隔离 Wall —— 不可移动、不可攻击的静态障碍物
 */

#pragma once

#include "entity.hpp"

class Wall : public Entity {
protected:
    float m_width  = 0.f;
    float m_height = 0.f;

public:
    Wall() = default;

    Wall(float x, float y, float width, float height)
        : Entity(EntityType::Wall, x, y, 0.f)
        , m_width(width)
        , m_height(height)
    {}

    // --- 访问器 ---
    float getWidth()  const { return m_width; }
    float getHeight() const { return m_height; }

    /// 获取墙壁的轴对齐矩形范围
    struct Rect { float left, top, width, height; };
    Rect getRect() const {
        return { m_x - m_width * 0.5f, m_y - m_height * 0.5f, m_width, m_height };
    }

    bool blocksMovement() const { return true; }
    bool blocksVision()   const { return true; }
};
