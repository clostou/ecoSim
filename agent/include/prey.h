/**
 * @file prey.h
 * @brief 猎物类定义
 */

#include "entity.h"


class Prey : public Animal
{
    Prey() : Animal()
    {
        m_id = EntityType::Prey;
    }
};