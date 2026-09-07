"""
Стохастические методы оптимизации, не требующие непрерывности функции:
имитация отжига (Simulated Annealing) и генетический алгоритм (Genetic
Algorithm). Обе реализации совместимы с ConstructiveNumber -- сравнение
значений функции идёт через оператор `<`, который для ConstructiveNumber
уже определён (сравнение по середине интервала, см. лабу 1).
"""

import math
import random


def _to_float(v):
    """Достаёт числовое значение для логов/остановки -- CN или float."""
    if hasattr(v, "a") and hasattr(v, "b"):
        return float((v.a + v.b) / 2)
    return float(v)


def simulated_annealing(black_box, x0, delta=0.5, t0=None, gamma=0.95,
                         max_iter=2000, seed=None):
    """
    Имитация отжига.

    black_box -- BlackBoxFunction (func + счётчик вызовов)
    x0        -- начальная точка (список float)
    delta     -- радиус, в котором ищем случайного "соседа" на каждом шаге
    t0        -- начальная температура (если None, оценивается как |f(x0)|)
    gamma     -- коэффициент остывания температуры (t *= gamma каждый шаг)
    max_iter  -- предел итераций
    """
    rng = random.Random(seed)
    black_box.reset_counters()

    x = list(x0)
    f_x = black_box.func(x)
    t = t0 if t0 is not None else max(abs(_to_float(f_x)), 1.0)

    best_x, best_f = list(x), f_x
    history = {"x": [], "f": [], "t": []}

    for it in range(max_iter):
        # случайный сосед в радиусе delta по каждой координате
        candidate = [xi + rng.uniform(-delta, delta) for xi in x]
        f_candidate = black_box.func(candidate)

        f_x_val = _to_float(f_x)
        f_cand_val = _to_float(f_candidate)

        if f_cand_val < f_x_val:
            accept = True
        else:
            # вероятность принять ухудшение (формула Метрополиса)
            p = math.exp((f_x_val - f_cand_val) / max(t, 1e-12))
            accept = rng.random() < p

        if accept:
            x, f_x = candidate, f_candidate
            if f_cand_val < _to_float(best_f):
                best_x, best_f = list(x), f_x

        t *= gamma

        history["x"].append([_to_float(xi) for xi in x])
        history["f"].append(_to_float(f_x))
        history["t"].append(t)

    return {
        "x_min": best_x,
        "f_min": _to_float(best_f),
        "iterations": max_iter,
        "func_calls": black_box.func_calls,
        "history": history,
    }


def _tournament_select(population, fitness, tournament_size, rng):
    """Турнирный отбор: берём случайных `tournament_size` особей, возвращаем лучшую."""
    idxs = rng.sample(range(len(population)), tournament_size)
    best_idx = min(idxs, key=lambda i: fitness[i])
    return population[best_idx]


def genetic_algorithm(black_box, bounds, pop_size=40, n_parents=20,
                       tournament_size=3, mutation_scale=0.3, mutation_rate=0.3,
                       max_iter=200, seed=None):
    """
    Генетический алгоритм.

    black_box       -- BlackBoxFunction
    bounds          -- список (low, high) для каждой координаты -- задаёт
                        область, где инициализируется случайная популяция
    pop_size        -- размер популяции (k)
    n_parents       -- сколько родителей отбираем на каждом поколении (r)
    tournament_size -- размер турнира для selection
    mutation_scale  -- масштаб гауссова шума при мутации
    mutation_rate   -- вероятность мутации каждой координаты потомка
    max_iter        -- число поколений
    """
    rng = random.Random(seed)
    black_box.reset_counters()
    dim = black_box.dim

    population = [
        [rng.uniform(lo, hi) for (lo, hi) in bounds] for _ in range(pop_size)
    ]
    fitness = [_to_float(black_box.func(ind)) for ind in population]

    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_x, best_f = list(population[best_idx]), fitness[best_idx]

    history = {"best_f": [], "mean_f": []}

    for gen in range(max_iter):
        # selection: отбираем n_parents родителей турнирным отбором
        parents = [
            _tournament_select(population, fitness, tournament_size, rng)
            for _ in range(n_parents)
        ]

        # crossover: новая популяция из случайных пар родителей + мутация
        new_population = []
        while len(new_population) < pop_size:
            p1, p2 = rng.sample(parents, 2)
            child = [(p1[d] + p2[d]) / 2 for d in range(dim)]
            for d in range(dim):
                if rng.random() < mutation_rate:
                    child[d] += rng.gauss(0, mutation_scale)
            new_population.append(child)

        population = new_population
        fitness = [_to_float(black_box.func(ind)) for ind in population]

        gen_best_idx = min(range(pop_size), key=lambda i: fitness[i])
        if fitness[gen_best_idx] < best_f:
            best_x, best_f = list(population[gen_best_idx]), fitness[gen_best_idx]

        history["best_f"].append(best_f)
        history["mean_f"].append(sum(fitness) / pop_size)

    return {
        "x_min": best_x,
        "f_min": best_f,
        "iterations": max_iter,
        "func_calls": black_box.func_calls,
        "history": history,
    }
