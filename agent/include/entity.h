/**
 * @file entity.h
 * @brief 生物实体类定义
 */

#pragma once

#include "config.h"


/// @brief 实体类型枚举
enum EntityType : uint8_t
{
    Wall = 0,		// 墙体
    Plant,			// 植物
    Predator,		// 捕食者
    Prey,			// 猎物
    
    Count,          // 实体类型数量

	Default = 100,	// Wall
	Creature,		// Plant
	Animal			// Predator, Prey
};


/// @brief 物理实体基类
///
/// 包含实体的基本物理属性
struct Entity
{
	Entity() :
		m_id(EntityType::Default),
        m_x(RNG::rand() * WORLD_WIDTH_HALF), m_y(RNG::rand() * WORLD_HEIGHT_HALF)
	{}

	Entity(EntityType type, float x, float y) :
		m_id(type), m_x(x), m_y(y)
	{}

	explicit Entity(Entity& other)
	{
		m_id = other.m_id;
		m_x = other.m_x + Conf::ENTITY_NEW_DX;
		m_y = other.m_y + Conf::ENTITY_NEW_DY;
		other.m_x -= Conf::ENTITY_NEW_DX;
		other.m_y -= Conf::ENTITY_NEW_DY;
	}

	/**
	 * @brief 更新实体状态
	 * 
	 * 主要是物理坐标的边界检查
	 */
	void update();

	uint8_t m_id;		/// 实体类型或ID
	float m_x;			/// x坐标
	float m_y;			/// y坐标

protected:
	static constexpr float WORLD_WIDTH_HALF = Conf::WORLD_WIDTH / 2.f;
	static constexpr float WORLD_HEIGHT_HALF = Conf::WORLD_HEIGHT / 2.f;
};


/// @brief 生物类
///
/// 相比物理实体，引入了生物活性变化、繁殖新个体等特性
struct Creature : public Entity
{
	Creature() :
		Entity(), m_reserve(Conf::RESERVE_MAX), m_breed(0), m_age(0)
	{
		m_id = EntityType::Creature;
	}

	Creature(Creature& other) :
		Entity(other), m_reserve(RESERVE_INIT), m_breed(0), m_age(0)
	{
		other.m_breed = 0;
	}

	/**
	 * @brief 更新生物状态
	 * 
	 * 包括储备值、繁育值和年龄的更新
	 * @param[in] delta_reserve 储备值的增量，正值表示能量获取，负值表示能量消耗或转移
	 */
	void update(int16_t delta_reserve);

	/**
	 * @brief 判断生物是否存活
	 */
	inline bool isAlive() { return m_reserve > 0 && m_age < Conf::AGE_MAX; }

	/**
	 * @brief 判断生物是否可以繁殖
	 */
	inline bool isBreed() { return m_breed >= Conf::BREED_MAX; }

	int16_t m_reserve;		/// 储备值，表示生物（生物质）的活性
	int16_t m_breed;		/// 繁育值，达到最大值时可以繁殖新个体
	int32_t m_age;			/// 年龄，表示生物的生存时间

protected:
	static constexpr int16_t RESERVE_INIT = Conf::TRANSFER_COEF * Conf::BREED_MAX;
};


/// @brief 动物类
///
/// 相比生物类，引入了自主运动、攻击等特性
struct Animal : public Creature
{
	Animal() :
		Creature(), m_health(Conf::HEALTH_MAX), m_energy(Conf::ENERGY_MAX),
		m_vel_x(0.f), m_vel_y(0.f)
	{
		m_id = EntityType::Animal;
	}

	Animal(Animal& other) :
		Creature(other), m_health(Conf::HEALTH_MAX), m_energy(Conf::ENERGY_MAX),
		m_vel_x(other.m_vel_x), m_vel_y(other.m_vel_y)
	{}

	Animal(Animal& first, Animal& second) :
		Animal()
	{
		if (first.m_id != second.m_id)
			m_health = 0;
		else
			m_id = first.m_id;
	}

	/**
	 * @brief 更新动物状态
	 * 
	 * 包括生命值、能量值、速度和位置的更新
	 * @param[in] delta_reserve 储备值的增量，正值表示能量获取，负值表示能量消耗或转移
	 */
	void update(int16_t delta_reserve);

	/**
	 * @brief 攻击其他动物
	 * @param[in] other 目标动物
	 */
	inline void attack(Animal& other)
	{
		if (m_is_dead || other.m_is_dead)
			return;

		m_energy -= 2;
		other.m_health -= Conf::ATK_DEFAULT;
	}

	int16_t m_health;			/// 生命值，归零后死亡
	int16_t m_energy;			/// 能量值，用于运动、攻击等行为的消耗
	bool m_is_dead = false;		/// 死亡标志
	float m_vel_x;				/// x轴速度分量
	float m_vel_y;				/// y轴速度分量

protected:
	static constexpr float LIMIT_FACTOR = 0.3f;    /// 低阈值系数
	static constexpr int16_t RESERVE_LOW = LIMIT_FACTOR * Conf::RESERVE_MAX;
	static constexpr int16_t ENERGY_LOW = LIMIT_FACTOR * Conf::ENERGY_MAX;
};
