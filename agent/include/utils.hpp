/**
 * @file utils.hpp
 * @brief 工具函数与基础数据结构
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>

// ---------------------------------------------------------------------------
// Vec2f — 二维浮点向量
// ---------------------------------------------------------------------------
struct Vec2f {
    float x = 0.f;
    float y = 0.f;

    Vec2f() = default;
    Vec2f(float x_, float y_) : x(x_), y(y_) {}

    Vec2f operator+(const Vec2f& o) const { return {x + o.x, y + o.y}; }
    Vec2f operator-(const Vec2f& o) const { return {x - o.x, y - o.y}; }
    Vec2f operator*(float s)       const { return {x * s, y * s}; }

    Vec2f& operator+=(const Vec2f& o) { x += o.x; y += o.y; return *this; }
    Vec2f& operator-=(const Vec2f& o) { x -= o.x; y -= o.y; return *this; }
    Vec2f& operator*=(float s)        { x *= s;   y *= s;   return *this; }

    float length()   const { return std::sqrt(x * x + y * y); }
    float lengthSq() const { return x * x + y * y; }

    Vec2f normalized() const {
        float len = length();
        if (len < 1e-8f) return {0.f, 0.f};
        return {x / len, y / len};
    }

    float dot(const Vec2f& o) const { return x * o.x + y * o.y; }

    static float distance(const Vec2f& a, const Vec2f& b) {
        return (a - b).length();
    }

    static float distanceSq(const Vec2f& a, const Vec2f& b) {
        return (a - b).lengthSq();
    }
};

inline Vec2f operator*(float s, const Vec2f& v) { return v * s; }

// ---------------------------------------------------------------------------
// 随机数工具
// ---------------------------------------------------------------------------
namespace RNG {

inline std::mt19937& engine() {
    static std::mt19937 gen(static_cast<unsigned>(std::time(nullptr)));
    return gen;
}

inline void seed(unsigned s) {
    engine().seed(s);
}

/// 返回 [0, 1) 的均匀随机浮点数
inline float uniform01() {
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(engine());
}

/// 返回 [lo, hi) 的均匀随机浮点数
inline float uniform(float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(engine());
}

/// 返回 N(mean, stddev) 的高斯随机浮点数
inline float gaussian(float mean, float stddev) {
    std::normal_distribution<float> dist(mean, stddev);
    return dist(engine());
}

/// 返回 [lo, hi] 的均匀随机整数
inline int randInt(int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(engine());
}

} // namespace RNG
