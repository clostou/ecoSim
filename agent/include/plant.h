/**
 * @file plant.h
 * @brief 植物类定义
 */

#include "entity.h"


class Plant : public Creature
{
	Plant() : Creature()
	{
		m_id = EntityType::Plant;
	}

	friend class DefaultPlantUpdater;
};


