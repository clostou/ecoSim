/**
 * 
 *  智能体配置文件
 * 
*/

#pragma once

#include <cstdint>

#include "utils.hpp"


struct DefaultAgentConfig
{
    static constexpr float WORLD_WIDTH = 1000;
    static constexpr float WORLD_HEIGHT = 1000;
	static float ENTITY_DIAMETER;
	static float ENTITY_NEW_DX;
	static float ENTITY_NEW_DY;

	static constexpr int16_t RESERVE_MAX = 500;
	static constexpr int16_t BREED_MAX = 300;
	static constexpr int16_t AGE_MAX = 1 << 9;
	
	static constexpr float TRANSFER_COEF = 0.8f;		/// 能量转移系数

	static constexpr int16_t HEALTH_MAX = 100;
	static constexpr int16_t ENERGY_MAX = 200;
	static constexpr int16_t ENERGY_CONSUME_MAX = 10;
	static constexpr int16_t ATK_DEFAULT = 20;

	static constexpr int16_t OBSERVATION_COUNT = 20;

	static inline void updateRandomState()
	{
		ENTITY_NEW_DX = RNG::rand();
		ENTITY_NEW_DY = RNG::rand();
	}
};


float DefaultAgentConfig::ENTITY_DIAMETER = 5.f;
float DefaultAgentConfig::ENTITY_NEW_DX = 0.f;
float DefaultAgentConfig::ENTITY_NEW_DY = 0.f;


using Conf = DefaultAgentConfig;


