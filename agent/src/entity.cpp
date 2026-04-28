/**
 * @file entity.cpp
 * @brief 生物实体类实现
 */

#include "entity.h"


void Entity::update()
{
	if (m_x < -WORLD_WIDTH_HALF)
		m_x = WORLD_WIDTH_HALF;
	else if (m_x > WORLD_WIDTH_HALF)
		m_x = -WORLD_WIDTH_HALF;
	
	if (m_y < -WORLD_HEIGHT_HALF)
		m_y = WORLD_HEIGHT_HALF;
	else if (m_y > WORLD_HEIGHT_HALF)
		m_y = -WORLD_HEIGHT_HALF;
}


void Creature::update(int16_t delta_reserve)
{
	Entity::update();

	if (! isAlive())
		return;

	if (delta_reserve > 0) {
		m_reserve += (int16_t)(Conf::TRANSFER_COEF * delta_reserve);
		if (m_reserve > Conf::RESERVE_MAX) {
			m_breed += (m_reserve - Conf::RESERVE_MAX);
			if (m_breed > Conf::BREED_MAX)
				m_breed = Conf::BREED_MAX;
			m_reserve = Conf::RESERVE_MAX;
		}
	}
	else {
		m_reserve += delta_reserve;
	}

	m_age++;
}


void Animal::update(int16_t delta_reserve)
{
	if (m_is_dead)
		return;

	Creature::update(delta_reserve);

	// 检查生命值
	if (m_health <= 0) {
		m_reserve = (int16_t)(Conf::TRANSFER_COEF * m_reserve);
		m_is_dead = true;
		return;
	}

	// 状态值更新
	if (m_reserve > RESERVE_LOW) {
		if (m_health < Conf::HEALTH_MAX) {
			m_health += 1;
			m_reserve -= 1;
		}
		if (m_energy < Conf::ENERGY_MAX) {
			m_energy += 1;
			m_reserve -= 1;
		}
	}

	// 速度计算
	float vel_quad = m_vel_x * m_vel_x + m_vel_y * m_vel_y;
	if (vel_quad > 1.f) {
		float inv_sqrt = invSqrt(vel_quad);
		m_vel_x *= inv_sqrt;
		m_vel_y *= inv_sqrt;
		vel_quad = 1.f;
	}

	// 位置更新
	if (m_energy > 0) {
		if (m_energy > ENERGY_LOW) {
			m_x += m_vel_x;
			m_y += m_vel_y;
			m_energy -= (int16_t)(vel_quad * Conf::ENERGY_CONSUME_MAX);
		}
		else {
			m_x += LIMIT_FACTOR * m_vel_x;
			m_y += LIMIT_FACTOR * m_vel_y;
			m_energy -= (int16_t)(LIMIT_FACTOR * vel_quad * Conf::ENERGY_CONSUME_MAX);
		}
	}
}
