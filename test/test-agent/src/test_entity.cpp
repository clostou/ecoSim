/**
 * @file test_entity.cpp
 * @brief 实体系统单元测试 —— Phase 1 核心数据结构验证
 *
 * 测试内容：
 *   1. Vec2f 基础运算
 *   2. Entity/Creature/Plant/Food/Wall/Animal/Predator/Prey 的创建与属性
 *   3. 攻击与存活判定
 *   4. 植物生长、食物衰败
 *   5. 动物能量更新、繁殖判定
 *   6. Observation 数据结构
 *   7. civ::Vector 增删查遍历
 *   8. EntityManager 综合功能
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>

#include "entity.hpp"
#include "creature.hpp"
#include "plant.hpp"
#include "food.hpp"
#include "wall.hpp"
#include "animal.hpp"
#include "predator.hpp"
#include "prey.hpp"
#include "observe.hpp"
#include "entity_manager.hpp"

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) \
    do { \
        ++test_count; \
        std::cout << "  [TEST] " << name << " ... "; \
    } while(0)

#define PASS() \
    do { \
        ++pass_count; \
        std::cout << "PASS" << std::endl; \
    } while(0)

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::cout << "FAIL at " << __FILE__ << ":" << __LINE__ \
                      << " (" #cond ")" << std::endl; \
            return; \
        } \
    } while(0)

#define CHECK_NEAR(a, b, eps) \
    CHECK(std::fabs((a) - (b)) < (eps))

// ========================================================================
// 1. Vec2f 测试
// ========================================================================
void test_vec2f() {
    std::cout << "[Vec2f]" << std::endl;

    TEST("construction");
    Vec2f v1(3.f, 4.f);
    CHECK(v1.x == 3.f && v1.y == 4.f);
    PASS();

    TEST("addition");
    Vec2f v2 = v1 + Vec2f(1.f, 2.f);
    CHECK_NEAR(v2.x, 4.f, 1e-6f);
    CHECK_NEAR(v2.y, 6.f, 1e-6f);
    PASS();

    TEST("subtraction");
    Vec2f v3 = v2 - v1;
    CHECK_NEAR(v3.x, 1.f, 1e-6f);
    CHECK_NEAR(v3.y, 2.f, 1e-6f);
    PASS();

    TEST("scalar multiply");
    Vec2f v4 = v1 * 2.f;
    CHECK_NEAR(v4.x, 6.f, 1e-6f);
    CHECK_NEAR(v4.y, 8.f, 1e-6f);
    PASS();

    TEST("length");
    CHECK_NEAR(v1.length(), 5.f, 1e-6f);
    PASS();

    TEST("normalized");
    Vec2f vn = v1.normalized();
    CHECK_NEAR(vn.length(), 1.f, 1e-5f);
    PASS();

    TEST("dot product");
    CHECK_NEAR(Vec2f(1, 0).dot(Vec2f(0, 1)), 0.f, 1e-6f);
    PASS();

    TEST("distance");
    CHECK_NEAR(Vec2f::distance(Vec2f(0, 0), Vec2f(3, 4)), 5.f, 1e-6f);
    PASS();
}

// ========================================================================
// 2. Entity 测试
// ========================================================================
void test_entity() {
    std::cout << "[Entity]" << std::endl;

    TEST("default construction");
    Entity e;
    CHECK(e.getType() == EntityType::Wall);
    CHECK(e.isAlive());
    PASS();

    TEST("parameterized construction");
    Entity e2(EntityType::Plant, 10.f, 20.f, 5.f);
    CHECK(e2.getType() == EntityType::Plant);
    CHECK_NEAR(e2.getX(), 10.f, 1e-6f);
    CHECK_NEAR(e2.getY(), 20.f, 1e-6f);
    CHECK_NEAR(e2.getRadius(), 5.f, 1e-6f);
    PASS();

    TEST("setPos and kill");
    e2.setPos(100.f, 200.f);
    CHECK_NEAR(e2.getX(), 100.f, 1e-6f);
    e2.kill();
    CHECK(!e2.isAlive());
    PASS();
}

// ========================================================================
// 3. Creature 测试
// ========================================================================
void test_creature() {
    std::cout << "[Creature]" << std::endl;

    TEST("construction");
    Creature c(EntityType::Plant, 5.f, 5.f, 3.f, 100.f, 300.f);
    CHECK_NEAR(c.getHealth(), 100.f, 1e-6f);
    CHECK_NEAR(c.getMaxHealth(), 100.f, 1e-6f);
    CHECK_NEAR(c.getAge(), 0.f, 1e-6f);
    CHECK(c.isAlive());
    PASS();

    TEST("beAttacked");
    c.beAttacked(30.f);
    CHECK_NEAR(c.getHealth(), 70.f, 1e-6f);
    CHECK(c.isAlive());
    PASS();

    TEST("beAttacked to death");
    c.beAttacked(80.f);
    CHECK(!c.isAlive());
    CHECK_NEAR(c.getHealth(), 0.f, 1e-6f);
    PASS();

    TEST("age to death");
    Creature c2(EntityType::Food, 0.f, 0.f, 1.f, 50.f, 10.f);
    for (int i = 0; i < 11; ++i) c2.updateAge(1.f);
    CHECK(!c2.isAlive());
    PASS();
}

// ========================================================================
// 4. Plant 测试
// ========================================================================
void test_plant() {
    std::cout << "[Plant]" << std::endl;

    TEST("construction");
    Plant p(10.f, 20.f);
    CHECK(p.getType() == EntityType::Plant);
    CHECK_NEAR(p.getGrowth(), 0.f, 1e-6f);
    CHECK(p.isAlive());
    PASS();

    TEST("growth update");
    p.update(1.f, 1.f, 1.f);  // dt=1, full light, growth_mod=1
    CHECK(p.getGrowth() > 0.f);
    CHECK(p.getDensity() > 0.f);
    PASS();

    TEST("growth to full");
    for (int i = 0; i < 10; ++i) p.update(1.f, 1.f, 1.f);
    CHECK_NEAR(p.getGrowth(), 1.f, 1e-6f);
    PASS();

    TEST("canSpread");
    // Need to wait for spread cooldown
    for (int i = 0; i < 30; ++i) p.update(1.f, 1.f, 1.f);
    CHECK(p.canSpread());
    p.resetSpreadTimer();
    CHECK(!p.canSpread());
    PASS();

    TEST("beEaten");
    Plant p2(0.f, 0.f);
    for (int i = 0; i < 5; ++i) p2.update(1.f, 1.f, 1.f);
    float energy = p2.beEaten(20.f);
    CHECK(energy > 0.f);
    CHECK(p2.getHealth() < AgentConfig::PLANT_HEALTH_MAX);
    PASS();
}

// ========================================================================
// 5. Food 测试
// ========================================================================
void test_food() {
    std::cout << "[Food]" << std::endl;

    TEST("construction");
    Food f(5.f, 5.f, 40.f);
    CHECK(f.getType() == EntityType::Food);
    CHECK_NEAR(f.getEnergy(), 40.f, 1e-6f);
    PASS();

    TEST("decay");
    f.update(1.f);
    CHECK(f.getEnergy() < 40.f);
    CHECK(f.isAlive());
    PASS();

    TEST("consume");
    float gained = f.consume(10.f);
    CHECK(gained > 0.f);
    PASS();

    TEST("decay to death");
    Food f2(0.f, 0.f, 1.f);
    for (int i = 0; i < 100; ++i) f2.update(1.f);
    CHECK(!f2.isAlive());
    PASS();
}

// ========================================================================
// 6. Wall 测试
// ========================================================================
void test_wall() {
    std::cout << "[Wall]" << std::endl;

    TEST("construction");
    Wall w(100.f, 200.f, 50.f, 30.f);
    CHECK(w.getType() == EntityType::Wall);
    CHECK_NEAR(w.getWidth(), 50.f, 1e-6f);
    CHECK_NEAR(w.getHeight(), 30.f, 1e-6f);
    PASS();

    TEST("rect");
    auto rect = w.getRect();
    CHECK_NEAR(rect.left, 75.f, 1e-6f);
    CHECK_NEAR(rect.top, 185.f, 1e-6f);
    PASS();

    TEST("blocks");
    CHECK(w.blocksMovement());
    CHECK(w.blocksVision());
    PASS();
}

// ========================================================================
// 7. Animal / Predator / Prey 测试
// ========================================================================
void test_animal() {
    std::cout << "[Animal / Predator / Prey]" << std::endl;

    TEST("Predator construction");
    Predator pred(50.f, 50.f);
    CHECK(pred.getType() == EntityType::Predator);
    CHECK_NEAR(pred.getAtk(), AgentConfig::PRED_ATK, 1e-6f);
    CHECK(pred.isAlive());
    PASS();

    TEST("Prey construction");
    Prey prey(60.f, 60.f);
    CHECK(prey.getType() == EntityType::Prey);
    CHECK_NEAR(prey.getSpeedMax(), AgentConfig::PREY_SPEED_MAX, 1e-6f);
    PASS();

    TEST("velocity and movement");
    pred.setVelocity(1.f, 0.f);
    float old_x = pred.getX();
    pred.applyVelocity(1.f);
    CHECK_NEAR(pred.getX(), old_x + 1.f, 1e-6f);
    PASS();

    TEST("speed clamping");
    pred.setVelocity(100.f, 100.f);
    Vec2f vel = pred.getVelocity();
    float speed = vel.length();
    CHECK(speed <= AgentConfig::PRED_SPEED_MAX + 1e-4f);
    PASS();

    TEST("attack prey");
    float prey_hp_before = prey.getHealth();
    pred.attackTarget(prey);
    CHECK(prey.getHealth() < prey_hp_before);
    PASS();

    TEST("predator eat food");
    Food food(0.f, 0.f, 50.f);
    float pred_energy_before = pred.getEnergy();
    pred.eatFood(food);
    CHECK(pred.getEnergy() >= pred_energy_before);  // may be equal if already full
    PASS();

    TEST("prey eat plant");
    Plant plant(0.f, 0.f);
    for (int i = 0; i < 5; ++i) plant.update(1.f, 1.f, 1.f);
    Prey prey2(0.f, 0.f);
    // Consume some energy first
    prey2.updateEnergy(10.f);
    float prey_energy_before = prey2.getEnergy();
    prey2.eatPlant(plant);
    CHECK(prey2.getEnergy() >= prey_energy_before);
    PASS();

    TEST("getState");
    auto st = pred.getState();
    CHECK(st.size() == 4);
    PASS();

    TEST("energy update");
    Predator p2(0.f, 0.f);
    float e_before = p2.getEnergy();
    p2.updateEnergy(1.f);
    CHECK(p2.getEnergy() < e_before);
    PASS();

    TEST("breed condition");
    // Fresh predator: breed_value=0, should not breed
    Predator p3(0.f, 0.f);
    CHECK(!p3.canBreed());
    PASS();
}

// ========================================================================
// 8. Observation 测试
// ========================================================================
void test_observation() {
    std::cout << "[Observation]" << std::endl;

    TEST("clear");
    Observation obs;
    obs.valid_count = 5;
    obs.state[0] = 99.f;
    obs.clear();
    CHECK(obs.valid_count == 0);
    CHECK_NEAR(obs.state[0], 0.f, 1e-6f);
    PASS();

    TEST("entry type one-hot");
    ObserveEntry entry;
    entry.setType(EntityType::Predator);
    CHECK_NEAR(entry.type[static_cast<int>(EntityType::Predator)], 1.0f, 1e-6f);
    CHECK_NEAR(entry.type[static_cast<int>(EntityType::Plant)], 0.0f, 1e-6f);
    PASS();
}

// ========================================================================
// 9. civ::Vector 测试
// ========================================================================
void test_civ_vector() {
    std::cout << "[civ::Vector]" << std::endl;

    TEST("emplace_back and size");
    civ::Vector<Plant> plants;
    civ::ID id1 = plants.emplace_back(10.f, 20.f);
    civ::ID id2 = plants.emplace_back(30.f, 40.f);
    CHECK(plants.size() == 2);
    PASS();

    TEST("access by ID");
    CHECK_NEAR(plants[id1].getX(), 10.f, 1e-6f);
    CHECK_NEAR(plants[id2].getX(), 30.f, 1e-6f);
    PASS();

    TEST("erase");
    plants.erase(id1);
    CHECK(plants.size() == 1);
    CHECK_NEAR(plants[id2].getX(), 30.f, 1e-6f);
    PASS();

    TEST("iteration after erase");
    int count = 0;
    for (auto& p : plants) {
        CHECK_NEAR(p.getX(), 30.f, 1e-6f);
        ++count;
    }
    CHECK(count == 1);
    PASS();

    TEST("slot reuse after erase");
    civ::ID id3 = plants.emplace_back(50.f, 60.f);
    CHECK(plants.size() == 2);
    CHECK_NEAR(plants[id3].getX(), 50.f, 1e-6f);
    PASS();

    TEST("remove_if");
    civ::Vector<Food> foods;
    foods.emplace_back(0.f, 0.f, 10.f);
    foods.emplace_back(0.f, 0.f, 0.5f);
    foods.emplace_back(0.f, 0.f, 20.f);
    foods.remove_if([](const Food& f) { return f.getEnergy() < 1.f; });
    CHECK(foods.size() == 2);
    PASS();

    TEST("getSlotAt");
    civ::Vector<Wall> walls;
    civ::ID wid = walls.emplace_back(1.f, 2.f, 3.f, 4.f);
    auto slot = walls.getSlotAt(0);
    CHECK(slot.id == wid);
    CHECK_NEAR(slot.object->getWidth(), 3.f, 1e-6f);
    PASS();
}

// ========================================================================
// 10. EntityManager 测试
// ========================================================================
void test_entity_manager() {
    std::cout << "[EntityManager]" << std::endl;

    TEST("create entities");
    EntityManager mgr;
    civ::ID w1 = mgr.createWall(0.f, 0.f, 100.f, 10.f);
    civ::ID p1 = mgr.createPlant(50.f, 50.f);
    civ::ID f1 = mgr.createFood(20.f, 20.f, 30.f);
    civ::ID pred1 = mgr.createPredator(100.f, 100.f);
    civ::ID prey1 = mgr.createPrey(200.f, 200.f);
    CHECK(mgr.totalCount() == 5);
    PASS();

    TEST("countByType");
    CHECK(mgr.countByType(EntityType::Wall) == 1);
    CHECK(mgr.countByType(EntityType::Plant) == 1);
    CHECK(mgr.countByType(EntityType::Food) == 1);
    CHECK(mgr.countByType(EntityType::Predator) == 1);
    CHECK(mgr.countByType(EntityType::Prey) == 1);
    PASS();

    TEST("access by type");
    CHECK_NEAR(mgr.walls()[w1].getWidth(), 100.f, 1e-6f);
    CHECK_NEAR(mgr.plants()[p1].getX(), 50.f, 1e-6f);
    CHECK_NEAR(mgr.foods()[f1].getEnergy(), 30.f, 1e-6f);
    PASS();

    TEST("markForRemoval and removeMarked");
    mgr.markForRemoval({EntityType::Food, f1});
    CHECK(mgr.countByType(EntityType::Food) == 1);  // not yet removed
    mgr.removeMarked();
    CHECK(mgr.countByType(EntityType::Food) == 0);
    CHECK(mgr.totalCount() == 4);
    PASS();

    TEST("markForAddition and addPending");
    mgr.markForAddition(Plant(70.f, 70.f));
    mgr.markForAddition(Plant(80.f, 80.f));
    CHECK(mgr.countByType(EntityType::Plant) == 1);  // not yet added
    mgr.addPending();
    CHECK(mgr.countByType(EntityType::Plant) == 3);
    PASS();

    TEST("forEach iteration");
    int plant_count = 0;
    mgr.forEachPlant([&](civ::ID id, Plant& p) {
        (void)id;
        CHECK(p.getType() == EntityType::Plant);
        ++plant_count;
    });
    CHECK(plant_count == 3);
    PASS();

    TEST("removeDeadEntities");
    // Kill the predator
    mgr.predators()[pred1].beAttacked(9999.f);
    CHECK(!mgr.predators()[pred1].isAlive());
    mgr.removeDeadEntities();
    CHECK(mgr.countByType(EntityType::Predator) == 0);
    PASS();
}

// ========================================================================
// main
// ========================================================================
int main() {
    std::cout << "=== ecoSim Phase 1 Unit Tests ===" << std::endl << std::endl;

    test_vec2f();
    test_entity();
    test_creature();
    test_plant();
    test_food();
    test_wall();
    test_animal();
    test_observation();
    test_civ_vector();
    test_entity_manager();

    std::cout << std::endl;
    std::cout << "=== Results: " << pass_count << "/" << test_count << " passed ===" << std::endl;

    if (pass_count == test_count) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED!" << std::endl;
        return 1;
    }
}
