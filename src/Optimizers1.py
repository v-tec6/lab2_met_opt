import math
from src.ConstructiveNumber import ConstructiveNumber


def _to_float(v):
    """
    Достаёт числовое значение из ConstructiveNumber (середину интервала)
    или просто float -- нужно для вычисления нормы градиента и логов,
    независимо от того, работаем мы с обычными числами или с CN.
    """
    if isinstance(v, ConstructiveNumber):
        return float((v.a + v.b) / 2)
    return float(v)


def _extract_eps(x):
    """
    Средняя полуширина (эпсилон) по всем координатам x, если это
    ConstructiveNumber. Для обычных float возвращает 0 (нет неопределённости).
    """
    widths = []
    for xi in x:
        if isinstance(xi, ConstructiveNumber):
            widths.append(float((xi.b - xi.a) / 2))
        else:
            widths.append(0.0)
    return sum(widths) / len(widths)


def gradient_descent(black_box, x0, lr=0.01, max_iter=1000, tol=1e-6):
    """
    Градиентный спуск (метод 1-го порядка).

    black_box -- объект BlackBoxFunction (func + grad + счётчики вызовов)
    x0        -- начальная точка (список float или ConstructiveNumber)
    lr        -- learning rate (шаг)
    max_iter  -- предел итераций
    tol       -- порог нормы градиента для остановки

    Возвращает словарь с итоговой точкой, историей траектории (для графиков)
    и статистикой по числу итераций/вызовов -- всё нужно для Блока 4.
    """
    black_box.reset_counters()
    x = list(x0)

    history = {"x": [], "f": [], "grad_norm": [], "eps": []}
    grad_norm = float("inf")
    it = 0

    for it in range(max_iter):
        try:
            f_val = black_box.func(x)
            g = black_box.grad(x)
            grad_norm = math.sqrt(sum(_to_float(gi) ** 2 for gi in g))
        except OverflowError:
            # метод явно разошёлся (числа улетели за пределы float) -- останавливаемся
            grad_norm = float("inf")
            break

        history["x"].append([_to_float(xi) for xi in x])
        history["f"].append(_to_float(f_val))
        history["grad_norm"].append(grad_norm)
        history["eps"].append(_extract_eps(x))

        if grad_norm < tol:
            break

        x = [xi - lr * gi for xi, gi in zip(x, g)]

    return {
        "x_min": x,
        "f_min": history["f"][-1] if history["f"] else float("inf"),
        "iterations": it + 1,
        "func_calls": black_box.func_calls,
        "grad_calls": black_box.grad_calls,
        "converged": grad_norm < tol,
        "history": history,
    }


def nelder_mead(black_box, x0, step=1.0, max_iter=1000, tol=1e-8,
                 alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """
    Метод деформируемого многогранника (Нелдера-Мида), метод 0-го порядка --
    не использует градиент, только значения функции.

    alpha -- коэффициент отражения, gamma -- коэффициент растяжения,
    rho -- коэффициент сжатия, sigma -- коэффициент общего сжатия к лучшей точке.
    """
    black_box.reset_counters()
    n = len(x0)

    # Начальный симплекс: x0 и n точек, сдвинутых на step по каждой оси
    simplex = [list(x0)]
    for i in range(n):
        point = list(x0)
        point[i] = point[i] + step
        simplex.append(point)

    fvals = [black_box.func(p) for p in simplex]
    history = {"best_x": [], "best_f": [], "spread": []}
    it = 0
    spread = float("inf")

    for it in range(max_iter):
        order = sorted(range(len(simplex)), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        history["best_x"].append([_to_float(c) for c in simplex[0]])
        history["best_f"].append(_to_float(fvals[0]))
        spread = _to_float(fvals[-1]) - _to_float(fvals[0])
        history["spread"].append(spread)

        if spread < tol:
            break

        centroid = [sum(simplex[i][d] for i in range(n)) / n for d in range(n)]
        worst = simplex[-1]

        xr = [centroid[d] + alpha * (centroid[d] - worst[d]) for d in range(n)]
        fr = black_box.func(xr)

        if fr < fvals[0]:
            xe = [centroid[d] + gamma * (xr[d] - centroid[d]) for d in range(n)]
            fe = black_box.func(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = [centroid[d] + rho * (worst[d] - centroid[d]) for d in range(n)]
            fc = black_box.func(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                best = simplex[0]
                for i in range(1, len(simplex)):
                    simplex[i] = [best[d] + sigma * (simplex[i][d] - best[d]) for d in range(n)]
                    fvals[i] = black_box.func(simplex[i])

    return {
        "x_min": simplex[0],
        "f_min": history["best_f"][-1],
        "iterations": it + 1,
        "func_calls": black_box.func_calls,
        "grad_calls": black_box.grad_calls,
        "converged": spread < tol,
        "history": history,
    }
