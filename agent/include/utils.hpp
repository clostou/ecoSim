/**
 * @file utils.hpp
 * @brief 流程控制、随机数生成等工具函数
 */

#include <cstdlib>
#include <time.h>




/// @brief 随机数生成器
namespace RNG
{

/**
 * @brief 设置随机数种子
 * @param[in] s 随机数种子，若为0则使用当前时间作为种子
 */
void seed(unsigned int s)
{
	static unsigned int _s = 1;
	if (s == NULL)
		std::srand(time(NULL));
	else if (_s != s) {
		std::srand(s);
		_s = s;
	}
}

/**
 * @brief 生成一个[-1, 1)范围内的随机浮点数
 */
inline float rand()
{
	return 2.f * (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 1.f;
}
	
	
} // namespace RNG


/**
 * @brief 快速求倒数平方根
 * 
 * 输入一个浮点数，返回其倒数平方根的近似值。该算法基于牛顿迭代法和位运算，单次迭代下相对误差小于0.1%。
 * 参考：https://en.wikipedia.org/wiki/Fast_inverse_square_root
 */
float invSqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) & y;							// evil floating point bit level hacking（邪恶的浮点数位运算黑科技）
	i  = 0x5f3759df - ( i >> 1 );					// what the fuck?（这是什么鬼？）
	y  = * ( float * ) & i;
	y  = y * ( threehalfs - ( x2 * y * y ) );		// 1st iteration （第一次迭代）
	// y  = y * ( threehalfs - ( x2 * y * y ) );	// 2nd iteration, this can be removed（第二次迭代，可以删除）

	return y;
}


