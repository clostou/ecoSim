#include "plant.h"


class DefaultPlantUpdater
{
	void operator()(Plant& plant)
	{
		plant.m_reserve += Conf::TRANSFER_COEF * Conf::ENERGY_MAX;
		if (plant.m_reserve > Conf::RESERVE_MAX)
			plant.m_reserve = Conf::RESERVE_MAX;
	}

	float getReserve(const Plant& plant) const
	{
		return plant.m_reserve / static_cast<float>(Conf::RESERVE_MAX);
	}
};


