class BlackBoxFunction:
    """
    Унифицированный интерфейс вызова целевой функции.

    - func(x)  -- значение функции в точке x (x -- список/кортеж длины dim)
    - grad(x)  -- градиент (аналитический) в точке x
    - Считает количество вызовов func и grad -- нужно для Блока 4
      (исследование числа вызовов функции/производной).

    Совместим с ConstructiveNumber "бесплатно": реализация func_impl/grad_impl
    использует только перегруженные операторы (+, -, *), поэтому работает
    одинаково что для float, что для ConstructiveNumber -- никакого
    дублирования кода не требуется.
    """

    def __init__(self, name, dim, func_impl, grad_impl):
        self._name = name
        self.dim = dim
        self._func_impl = func_impl
        self._grad_impl = grad_impl
        self.func_calls = 0
        self.grad_calls = 0

    def name(self):
        return self._name

    def func(self, x):
        assert len(x) == self.dim, f"{self._name}: ожидалось {self.dim} аргументов, получено {len(x)}"
        self.func_calls += 1
        return self._func_impl(x)

    def grad(self, x):
        assert len(x) == self.dim, f"{self._name}: ожидалось {self.dim} аргументов, получено {len(x)}"
        self.grad_calls += 1
        return self._grad_impl(x)

    def reset_counters(self):
        self.func_calls = 0
        self.grad_calls = 0

    def __repr__(self):
        return f"BlackBoxFunction({self._name}, dim={self.dim})"


def _make_diagonal_quadratic(name, weights):
    """
    f(x) = 0.5 * sum(w_i * x_i^2)   -- диагональная квадратичная форма.
    Число обусловленности = max(weights) / min(weights).
    grad_i = w_i * x_i
    """
    dim = len(weights)

    def f(x):
        return 0.5 * sum(w * (xi * xi) for w, xi in zip(weights, x))

    def g(x):
        return [w * xi for w, xi in zip(weights, x)]

    return BlackBoxFunction(name, dim, f, g)


# --- Блок 2: три чёрных ящика ---

# 1) Квадратичная, 6-арная, число обусловленности = 1 (все веса равны)
QuadraticGood = _make_diagonal_quadratic(
    "Quadratic 6D (cond=1)", weights=[1, 1, 1, 1, 1, 1]
)

# 2) Квадратичная, 4-арная, число обусловленности = 100
QuadraticBad = _make_diagonal_quadratic(
    "Quadratic 4D (cond=100)", weights=[1, 10, 30, 100]
)


def _rosenbrock_func(x):
    x1, x2, x3 = x
    t1 = x2 - x1 * x1
    t2 = x3 - x2 * x2
    term1 = 100 * (t1 * t1) + (1 - x1) * (1 - x1)
    term2 = 100 * (t2 * t2) + (1 - x2) * (1 - x2)
    return term1 + term2


def _rosenbrock_grad(x):
    x1, x2, x3 = x
    g1 = (-400) * x1 * (x2 - x1 * x1) - 2 * (1 - x1)
    g2 = 200 * (x2 - x1 * x1) - 400 * x2 * (x3 - x2 * x2) - 2 * (1 - x2)
    g3 = 200 * (x3 - x2 * x2)
    return [g1, g2, g3]


# 3) Функция Розенброка, 3-арная
Rosenbrock = BlackBoxFunction("Rosenbrock 3D", 3, _rosenbrock_func, _rosenbrock_grad)
