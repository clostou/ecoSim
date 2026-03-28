/**
 * 
 *   智能体的对外观测
 * 
*/


template <typename T, int N>
struct Observe
{
	Observe() = delete;
	Observe(const Observe&) = delete;

	Observe(float x, float y) :
		x(x), y(y)
	{}

	float x, y;

};


