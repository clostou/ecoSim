/**
 * 
 *   智能体配置文件
 * 
*/

#pragma once

#include <cstdint>

#include "utils.hpp"


struct DefaultAgentConfig
{
	static constexpr int8_t HEALTH_MAX = 100;
	static constexpr float ENERGE_MAX = 20;
	static constexpr float REVERSE_MAX = 20;
	static constexpr float ATK_DEFAULT = 20;
	static float ENTITY_DIAMETER;
	static float ENTITY_NEW_DX;
	static float ENTITY_NEW_DY;

	static void updateRandomState()
	{
		ENTITY_NEW_DX = rand();
		ENTITY_NEW_DY = rand();
	}
};


float DefaultAgentConfig::ENTITY_DIAMETER = 5.f;
float DefaultAgentConfig::ENTITY_NEW_DX = 0.f;
float DefaultAgentConfig::ENTITY_NEW_DY = 0.f;


using Conf = DefaultAgentConfig;


