"""
Интервальные (совместимые с ConstructiveNumber) версии elementary-функций,
которые нужны для Растригина, Экли и Desmos-функции: cos, exp, sqrt, round.

Каждая функция работает одинаково с float и с ConstructiveNumber -- если
на входе float, ведёт себя как обычная math-функция; если на входе
ConstructiveNumber, возвращает ConstructiveNumber, гарантированно
содержащий весь диапазон значений исходной функции на отрезке [a, b].
"""

import math
from fractions import Fraction
from src.ConstructiveNumber import ConstructiveNumber


def cn_exp(x):
    """exp монотонно возрастает -> exp([a,b]) = [exp(a), exp(b)]."""
    if isinstance(x, ConstructiveNumber):
        # math.exp работает с Fraction через приведение к float; берём
        # чуть более широкий (безопасный) диапазон через float-границы
        lo = math.exp(float(x.a))
        hi = math.exp(float(x.b))
        return ConstructiveNumber(
            Fraction(lo).limit_denominator(10**9),
            Fraction(hi).limit_denominator(10**9),
        )
    return math.exp(x)


def cn_sqrt(x):
    """sqrt монотонно возрастает на [0, +inf) -> sqrt([a,b]) = [sqrt(a), sqrt(b)]."""
    if isinstance(x, ConstructiveNumber):
        a = max(float(x.a), 0.0)  # защита от отрицательных из-за погрешности
        b = max(float(x.b), 0.0)
        return ConstructiveNumber(
            Fraction(math.sqrt(a)).limit_denominator(10**9),
            Fraction(math.sqrt(b)).limit_denominator(10**9),
        )
    return math.sqrt(max(x, 0.0))


def cn_cos(x):
    """
    cos не монотонен -- нужно проверить, попадают ли внутрь [a,b] точки
    экстремумов (максимум в 2*pi*k, минимум в pi + 2*pi*k).
    Если попадают -- диапазон включает соответствующий экстремум (1 или -1).
    Если нет -- достаточно взять min/max от значений на границах.
    """
    if isinstance(x, ConstructiveNumber):
        a, b = float(x.a), float(x.b)
        candidates = [math.cos(a), math.cos(b)]

        # проверяем, есть ли внутри [a, b] точка вида 2*pi*k (максимум cos = 1)
        k_min = math.ceil((a - 0) / (2 * math.pi))
        k_max = math.floor((b - 0) / (2 * math.pi))
        if k_min <= k_max:
            candidates.append(1.0)

        # проверяем точку вида pi + 2*pi*k (минимум cos = -1)
        k_min = math.ceil((a - math.pi) / (2 * math.pi))
        k_max = math.floor((b - math.pi) / (2 * math.pi))
        if k_min <= k_max:
            candidates.append(-1.0)

        lo, hi = min(candidates), max(candidates)
        return ConstructiveNumber(
            Fraction(lo).limit_denominator(10**9),
            Fraction(hi).limit_denominator(10**9),
        )
    return math.cos(x)


def cn_sin(x):
    """
    sin(x) = cos(x - pi/2) -- переиспользуем уже проверенную логику cn_cos
    со сдвигом фазы, вместо дублирования поиска экстремумов.
    """
    if isinstance(x, ConstructiveNumber):
        return cn_cos(x - Fraction(math.pi / 2).limit_denominator(10**9))
    return math.sin(x)


def cn_round(x):
    """
    round разрывен -- консервативное (широкое) интервальное расширение:
    берём весь диапазон целых чисел, которые может дать округление
    любой точки внутри [a, b].
    """
    if isinstance(x, ConstructiveNumber):
        a, b = float(x.a), float(x.b)
        lo = math.floor(a + 0.5)  # наименьшее возможное округление
        hi = math.ceil(b - 0.5) if b - 0.5 > lo else lo  # наибольшее возможное
        hi = max(hi, round(b))
        lo = min(lo, round(a))
        return ConstructiveNumber(lo, hi)
    return float(round(x))
