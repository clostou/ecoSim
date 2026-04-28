/**
 * @file agent.h
 * @brief 包含控制动物运动的智能体类的定义
 */

#pragma once

#include "config.h"
#include "entity.h"
#include "utils.hpp"

#include <vector>


struct AgentObervation
{
    AgentObervation() = default;

    AgentObervation(const AgentObervation& other)
    {
        // 复制构造函数，根据需要实现
    }

    AgentObervation& operator=(const AgentObervation& other)
    {
        // 赋值运算符，根据需要实现
        return *this;
    }

    AgentObervation(AgentObervation&& other) noexcept
    {
        // 移动构造函数，根据需要实现
    }
};


struct Agent
{
    Agent() = default;

    virtual void update() = 0;

    AgentObervation (*getObservation)(const Entity& entity) = nullptr;

    std::vector<float> (*decideAction)(const AgentObervation& observation) = nullptr;
};


struct AgentArg
{
    AgentArg() = default;
};


