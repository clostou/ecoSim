#include <SFML/Graphics.hpp>
#include <random>
#include <algorithm>
#include "event.hpp"
#include "star.hpp"
#include "configuration.hpp"


std::vector<Star> createStars(uint32_t count)
{
    std::vector<Star> stars;
    stars.reserve(count);

    // Random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    // Define a star free zone
    sf::Vector2f const window_world_size = conf::window_size_f * conf::near;
    sf::FloatRect const star_free_zone = {-window_world_size * 0.5f, window_world_size};

    // Create randomly distributed stars on the screen
    for (uint32_t i{count}; i--; )
    {
        float const x = (2.0f * dis(gen) - 1.0f) * conf::window_size_f.x;
        float const y = (2.0f * dis(gen) - 1.0f) * conf::window_size_f.y;
        float const z = dis(gen) * (conf::far - conf::near) + conf::near;

        if (star_free_zone.contains({x, y}))
        {
            ++i;
            continue;
        }

        stars.push_back({{x, y}, z});
    }

    // Depth ordering
    std::sort(stars.begin(), stars.end(),
              [](Star const& s1, Star const& s2) { return s1.z < s2.z; });

    return stars;
}


void updateGeometry(uint32_t idx, Star const& s, sf::VertexArray& va)
{
    float const scale = 1.0f / s.z;
    float const depth_ratio = (s.z - conf::near) / (conf::far - conf::near);
    float const color_ratio = 1.0f - depth_ratio;
    auto const c = static_cast<uint8_t>(color_ratio * 255.0f);

    sf::Vector2f const p = s.position * scale;
    float const r = conf::radius * scale;
    uint32_t const i = 6 * idx;

    va[i + 0].position = {p.x - r, p.y - r};
    va[i + 1].position = {p.x + r, p.y - r};
    va[i + 2].position = {p.x - r, p.y + r};
    va[i + 3].position = {p.x + r, p.y - r};
    va[i + 4].position = {p.x - r, p.y + r};
    va[i + 5].position = {p.x + r, p.y + r};

    sf::Color const color{c, c, c};
    va[i + 0].color = color;
    va[i + 1].color = color;
    va[i + 2].color = color;
    va[i + 3].color = color;
    va[i + 4].color = color;
    va[i + 5].color = color;
}


void isOrdered(std::vector<Star>& stars, uint32_t index)
{
    bool order = true;
    float last = conf::far;
    for (uint32_t i{conf::count}; i--; ) {
        uint32_t j = (i + index) % conf::count;
        Star &s = stars[j];
        order = order && (last >= s.z);
        last = s.z;
    }
    printf("%d\n", order);
}


int main()
{
    auto window = sf::RenderWindow(sf::VideoMode({conf::window_size.x, conf::window_size.y}), "CMake SFML Project");
    window.setFramerateLimit(conf::max_framerate);
    window.setMouseCursorVisible(false);

    std::vector<Star> stars = createStars(conf::count);

    sf::VertexArray va{sf::PrimitiveType::Triangles, 6 * conf::count};

    // Pre-fill texture coords as they will remain constant
    sf::Texture texture;
    bool with_texture = false;
    if (texture.loadFromFile("res/star.png"))
    {
        texture.setSmooth(true);
        if (texture.generateMipmap())
        {
            auto const texture_size_f = static_cast<sf::Vector2f>(texture.getSize());
            for (uint32_t idx{conf::count}; idx--; )
            {
                uint32_t const i = 6 * idx;
                va[i + 0].texCoords = {0.0f, 0.0f};
                va[i + 1].texCoords = {texture_size_f.x, 0.0f};
                va[i + 2].texCoords = {0.0f, texture_size_f.y};
                va[i + 3].texCoords = {texture_size_f.x, 0.0f};
                va[i + 4].texCoords = {0.0f, texture_size_f.y};
                va[i + 5].texCoords = {texture_size_f.x, texture_size_f.y};
            }
            with_texture = true;
        }
    }

    uint32_t index = 0;
    while (window.isOpen())
    {
        processEvents(window);

        window.clear();

//        sf::CircleShape shape{conf::radius};
//        shape.setOrigin({conf::radius, conf::radius});

        uint32_t ind = index;
        for (uint32_t i{conf::count}; i--; )
        {
            uint32_t j = (i + index) % conf::count;
            Star& s = stars[j];

//            float const scale = 1.0f / s.z;
//            shape.setPosition({s.position.x * scale + conf::window_size_f.x * 0.5f,
//                               s.position.y * scale + conf::window_size_f.y * 0.5f});
//            shape.setScale({scale, scale});
//            float const depth_ratio = (s.z - conf::near) / (conf::far - conf::near);
//            float const color_ratio = 1.0f - depth_ratio;
//            auto const c = static_cast<uint8_t>(color_ratio * 255.0f);
//            shape.setFillColor({c, c, c});
//            window.draw(shape);

            updateGeometry(conf::count - (i + 1), s, va);

            s.z -= conf::speed * conf::dt;
            if (s.z <= conf::near)
            {
                s.z += conf::far - conf::near;
                ind += 1;
            }
        }

        index = ind % conf::count;

        if (with_texture)
        {
            sf::RenderStates states;
            states.transform.translate(conf::window_size_f * 0.5f);
            states.texture = &texture;
            window.draw(va, states);
        }
        else
        {
            sf::Transform tf;
            tf.translate(conf::window_size_f * 0.5f);
            window.draw(va, tf);
        }

        window.display();
    }
}
