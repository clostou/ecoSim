/**
 * 
 *   ¹¤¾ßº¯Êý
 * 
*/

#include <cstdlib>
#include <time.h>


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

inline float rand()
{
	return 2.f * (to<float>std::rand() / to<float>std::RAND_MAX) - 1.f;
}


