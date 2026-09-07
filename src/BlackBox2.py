"""
Три тестовые функции лабы 2, в том же формате BlackBoxFunction, что и в
лабе 1: func(x) + счётчик вызовов. Все реализации совместимы одновременно
с float и с ConstructiveNumber -- используют только перегруженные операторы
(+, -, *) и интервальные версии elementary-функций из IntervalMath.

Аналитический градиент сознательно не реализован -- по условию лабы
сравниваются именно стохастические методы, не требующие непрерывности
и производной.
"""

import math
from src.IntervalMath import cn_cos, cn_sin, cn_exp, cn_sqrt, cn_round


class BlackBoxFunction:
    def __init__(self, name, dim, func_impl, is_smooth):
        self._name = name
        self.dim = dim
        self._func_impl = func_impl
        self.is_smooth = is_smooth
        self.func_calls = 0
        self.grad_calls = 0

    def name(self):
        return self._name

    def func(self, x):
        assert len(x) == self.dim, f"{self._name}: ожидалось {self.dim} аргументов"
        self.func_calls += 1
        return self._func_impl(x)

    def reset_counters(self):
        self.func_calls = 0
        self.grad_calls = 0

    def __repr__(self):
        return f"BlackBoxFunction({self._name}, dim={self.dim})"


def _pow2(v):
    """Возведение в квадрат через умножение -- работает и для float, и для CN."""
    return v * v


PI = math.pi
A_RASTRIGIN = 10


def _rastrigin2_func(x):
    x1, x2 = x
    term1 = x1 * x1 - A_RASTRIGIN * cn_cos(2 * PI * x1)
    term2 = x2 * x2 - A_RASTRIGIN * cn_cos(2 * PI * x2)
    return 2 * A_RASTRIGIN + term1 + term2


Rastrigin2 = BlackBoxFunction("Rastrigin-2", 2, _rastrigin2_func, is_smooth=True)


def _ackley2_func(x):
    x1, x2 = x
    sum_sq = x1 * x1 + x2 * x2
    term1 = -20 * cn_exp(-0.2 * cn_sqrt(0.5 * sum_sq))
    cos_sum = 0.5 * (cn_cos(2 * PI * x1) + cn_cos(2 * PI * x2))
    term2 = -cn_exp(cos_sum)
    return term1 + term2 + math.e + 20


Ackley2 = BlackBoxFunction("Ackley-2", 2, _ackley2_func, is_smooth=True)


def _desmos_func(x):
    x1, x2 = x
    part1 = _pow2(x1 * (cn_round(cn_sin(10 * x2)) + 2))
    part1 = _pow2(part1 + x2 - 10)

    part2 = _pow2(x2 * (cn_round(cn_sin(7 * x1)) + 2))
    part2 = _pow2(x1 + part2 - 7)

    return part1 + part2


Desmos = BlackBoxFunction("Desmos (негладкая)", 2, _desmos_func, is_smooth=False)
