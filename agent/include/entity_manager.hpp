/**
 * @file entity_manager.hpp
 * @brief 实体管理器 EntityManager —— 基于 civ::Vector<T> 的高效实体生命周期管理
 */

#pragma once

#include "../../misc/index_vector.hpp"
#include "wall.hpp"
#include "plant.hpp"
#include "food.hpp"
#include "predator.hpp"
#include "prey.hpp"
#include <functional>

class EntityManager {
    // 按类型分存
    civ::Vector<Wall>      m_walls;
    civ::Vector<Plant>     m_plants;
    civ::Vector<Food>      m_foods;
    civ::Vector<Predator>  m_predators;
    civ::Vector<Prey>      m_preys;

    // 延迟删除队列
    std::vector<EntityRef> m_to_remove;

    // 延迟添加队列
    std::vector<Wall>      m_walls_to_add;
    std::vector<Plant>     m_plants_to_add;
    std::vector<Food>      m_foods_to_add;
    std::vector<Predator>  m_predators_to_add;
    std::vector<Prey>      m_preys_to_add;

public:
    // --- 立即创建实体 ---
    civ::ID createWall(float x, float y, float w, float h) {
        return m_walls.emplace_back(x, y, w, h);
    }

    civ::ID createPlant(float x, float y) {
        return m_plants.emplace_back(x, y);
    }

    civ::ID createFood(float x, float y, float energy) {
        return m_foods.emplace_back(x, y, energy);
    }

    civ::ID createPredator(float x, float y) {
        return m_predators.emplace_back(x, y);
    }

    civ::ID createPrey(float x, float y) {
        return m_preys.emplace_back(x, y);
    }

    // --- 延迟添加 ---
    void markForAddition(Wall&& w)      { m_walls_to_add.push_back(std::move(w)); }
    void markForAddition(Plant&& p)     { m_plants_to_add.push_back(std::move(p)); }
    void markForAddition(Food&& f)      { m_foods_to_add.push_back(std::move(f)); }
    void markForAddition(Predator&& p)  { m_predators_to_add.push_back(std::move(p)); }
    void markForAddition(Prey&& p)      { m_preys_to_add.push_back(std::move(p)); }

    // --- 延迟删除 ---
    void markForRemoval(EntityRef ref) {
        m_to_remove.push_back(ref);
    }

    // --- 执行延迟添加 ---
    void addPending() {
        for (auto& w : m_walls_to_add)      m_walls.push_back(w);
        for (auto& p : m_plants_to_add)     m_plants.push_back(p);
        for (auto& f : m_foods_to_add)      m_foods.push_back(f);
        for (auto& p : m_predators_to_add)  m_predators.push_back(p);
        for (auto& p : m_preys_to_add)      m_preys.push_back(p);
        m_walls_to_add.clear();
        m_plants_to_add.clear();
        m_foods_to_add.clear();
        m_predators_to_add.clear();
        m_preys_to_add.clear();
    }

    // --- 执行延迟删除 ---
    void removeMarked() {
        for (const auto& ref : m_to_remove) {
            switch (ref.type) {
                case EntityType::Wall:     m_walls.erase(ref.id);     break;
                case EntityType::Plant:    m_plants.erase(ref.id);    break;
                case EntityType::Food:     m_foods.erase(ref.id);     break;
                case EntityType::Predator: m_predators.erase(ref.id); break;
                case EntityType::Prey:     m_preys.erase(ref.id);     break;
                default: break;
            }
        }
        m_to_remove.clear();
    }

    // --- 清除已死亡的实体 ---
    void removeDeadEntities() {
        m_plants.remove_if([](const Plant& p)       { return !p.isAlive(); });
        m_foods.remove_if([](const Food& f)         { return !f.isAlive(); });
        m_predators.remove_if([](const Predator& p) { return !p.isAlive(); });
        m_preys.remove_if([](const Prey& p)         { return !p.isAlive(); });
    }

    // --- 按类型访问 civ::Vector ---
    civ::Vector<Wall>&      walls()      { return m_walls; }
    civ::Vector<Plant>&     plants()     { return m_plants; }
    civ::Vector<Food>&      foods()      { return m_foods; }
    civ::Vector<Predator>&  predators()  { return m_predators; }
    civ::Vector<Prey>&      preys()      { return m_preys; }

    const civ::Vector<Wall>&      walls()      const { return m_walls; }
    const civ::Vector<Plant>&     plants()     const { return m_plants; }
    const civ::Vector<Food>&      foods()      const { return m_foods; }
    const civ::Vector<Predator>&  predators()  const { return m_predators; }
    const civ::Vector<Prey>&      preys()      const { return m_preys; }

    // --- 按类型遍历（回调带 civ::ID） ---
    void forEachWall(std::function<void(civ::ID, Wall&)> fn) {
        for (uint64_t i = 0; i < m_walls.size(); ++i) {
            auto slot = m_walls.getSlotAt(i);
            fn(slot.id, *slot.object);
        }
    }

    void forEachPlant(std::function<void(civ::ID, Plant&)> fn) {
        for (uint64_t i = 0; i < m_plants.size(); ++i) {
            auto slot = m_plants.getSlotAt(i);
            fn(slot.id, *slot.object);
        }
    }

    void forEachFood(std::function<void(civ::ID, Food&)> fn) {
        for (uint64_t i = 0; i < m_foods.size(); ++i) {
            auto slot = m_foods.getSlotAt(i);
            fn(slot.id, *slot.object);
        }
    }

    void forEachPredator(std::function<void(civ::ID, Predator&)> fn) {
        for (uint64_t i = 0; i < m_predators.size(); ++i) {
            auto slot = m_predators.getSlotAt(i);
            fn(slot.id, *slot.object);
        }
    }

    void forEachPrey(std::function<void(civ::ID, Prey&)> fn) {
        for (uint64_t i = 0; i < m_preys.size(); ++i) {
            auto slot = m_preys.getSlotAt(i);
            fn(slot.id, *slot.object);
        }
    }

    // --- 计数 ---
    uint64_t countByType(EntityType type) const {
        switch (type) {
            case EntityType::Wall:     return m_walls.size();
            case EntityType::Plant:    return m_plants.size();
            case EntityType::Food:     return m_foods.size();
            case EntityType::Predator: return m_predators.size();
            case EntityType::Prey:     return m_preys.size();
            default: return 0;
        }
    }

    uint64_t totalCount() const {
        return m_walls.size() + m_plants.size() + m_foods.size()
             + m_predators.size() + m_preys.size();
    }
};
