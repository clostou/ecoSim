/**
 * @file wall.h
 * @brief 墙体类定义
 */

#include "entity.h"


class Wall : Entity
{
	Wall() : Entity()
    {
        m_id = EntityType::Wall;
    }
};


