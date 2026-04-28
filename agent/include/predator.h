/**
 * @file predator.h
 * @brief 捕食者类定义
 */

#include "entity.h"


class Predator : public Animal
{
    Predator() : Animal()
    {
        m_id = EntityType::Predator;
    }
};