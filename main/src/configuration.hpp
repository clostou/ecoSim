#pragma once


namespace conf
{
// Windows configuration
sf::Vector2u const window_size = {1280, 720};
sf::Vector2f const window_size_f = static_cast<sf::Vector2f>(window_size);
uint32_t const max_framerate = 144;
float const dt = 1.0f / static_cast<float>(max_framerate);

// Star configuration
uint32_t const count = 10'000;
float const radius = 10.0f;
float const speed = 0.2f;
float const far = 10.0f;  // greater than near
float const near = 0.1f;  // range in 0.0 ~ 2.0
}
