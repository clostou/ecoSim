#include "plant.h"


void Plant::update(float heal, float grow)
{
	m_health += heal;
	if (m_health > Conf::HEALTH_MAX)
		m_health = Conf::HEALTH_MAX;
	else if ()
}


