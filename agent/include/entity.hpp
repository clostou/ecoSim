/**
*
*   定义结构体
*
 */

#include "config.hpp"


enum class EntityType : uint8_t
{
	Default = 0,		// Wall
	Creature,		// Plant
	Animal			// Predator, Prey
};


class Entity
{
protected:
	uint8_t m_id;
	float m_x, m_y;

	int8_t m_health, m_breed, m_atk;
	float m_vel_x, m_vel_y;
	float m_energy, m_reverse;

public:
	Entity() :
		m_id(EntityT	ype::Default), m_x(0.f), m_y(0.f),
		m_health(Conf::HEALTH_MAX), m_breed(0), m_atk(Conf::ATK_DEFAULT),
		m_vel_x(0.f), m_vel_y(0.f), m_energe(Conf::ENERGE_MAX), m_reverse(Conf::REVERSE_MAX)
	{}

	Entity(EntityType type, float x, float y) :
		Entity(),
		m_id(type), m_x(x), x_y(y)
	{}

	explicit Entity(const Entity& other) :
		Entity()
	{
		m_id = other.m_id;
		m_x = other.m_x + Conf::ENTITY_NEW_DX;
		m_y = other.m_y + Conf::ENTITY_NEW_DY;
	}

	void move(float x, float y)
	{

	}
};


class Creature : public Entity
{
public:
	void beAttack(const Creature& other) { m_health -= other.m_atk; }

	inline bool isAlive() { return m_health > 0; }
};


class Animal : public Creature
{
public:
	Animal(const Animal& first, const Animal& second) :
		Animal()
	{
		if (first.m_id != second.m_id)
			m_health = 0;
		else
			m_id = first.m_id;
	}

	template <typename T>
	void update(const T& observe) {}

	inline void attack(const Creature& other) { other.beAttack(*this); }
};


