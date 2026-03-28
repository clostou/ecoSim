/**
* 
*   生产者（植物）
* 
*/

#include "entity.hpp"


class Plant : public Creature
{
	m_id = EntityType::Plant;

	void update(float heal, float grow);
};


