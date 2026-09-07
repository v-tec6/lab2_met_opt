from fractions import Fraction


class ConstructiveNumber:
    """
    Конструктивное число — представление вещественного числа
    через рациональный отрезок [a, b], гарантированно содержащий
    "истинное" значение.

    a, b -- границы отрезка, всегда Fraction (a <= b).
    """

    def __init__(self, a, b):
        """
        Основной конструктор: задаём число напрямую через границы a, b in Q.
        """
        a = Fraction(a).limit_denominator()
        b = Fraction(b).limit_denominator()
        self.a = min(a, b)
        self.b = max(a, b)

    @classmethod
    def from_center_eps(cls, x: float, eps: float):
        """
        Альтернативный конструктор: задаём число через центр x и погрешность eps.
        a = x - eps, b = x + eps
        """
        return cls(x - eps, x + eps)

    def __neg__(self):
        return ConstructiveNumber(-self.b, -self.a)

    def __repr__(self):
        return f"ConstructiveNumber[{self.a}, {self.b}]"

    @staticmethod
    def _as_cn(other):
        """
        Приводит обычное число (int/float/Fraction) к ConstructiveNumber
        как вырожденный интервал [c, c]. Нужно, чтобы одна и та же
        логика операций работала и для CN+CN, и для CN+число.
        """
        if isinstance(other, ConstructiveNumber):
            return other
        c = Fraction(other).limit_denominator()
        return ConstructiveNumber(c, c)

    # --- Сложение ---
    def __add__(self, other):
        other = self._as_cn(other)
        return ConstructiveNumber(self.a + other.a, self.b + other.b)

    __radd__ = __add__  # 5 + cn работает так же, как cn + 5

    # --- Вычитание ---
    def __sub__(self, other):
        other = self._as_cn(other)
        # минимум разности: из своего минимума вычитаем максимум другого
        return ConstructiveNumber(self.a - other.b, self.b - other.a)

    def __rsub__(self, other):
        # 5 - cn  =  (5 как CN) - cn
        return self._as_cn(other).__sub__(self)

    # --- Умножение ---
    def __mul__(self, other):
        other = self._as_cn(other)
        products = [
            self.a * other.a,
            self.a * other.b,
            self.b * other.a,
            self.b * other.b,
        ]
        return ConstructiveNumber(min(products), max(products))

    __rmul__ = __mul__  # 5 * cn работает так же, как cn * 5

    # --- Деление ---
    def __truediv__(self, other):
        other = self._as_cn(other)
        if other.a <= 0 <= other.b:
            raise ZeroDivisionError(
                "Делитель содержит 0 внутри своего интервала — деление не определено"
            )
        # обратный интервал: [1/b2, 1/a2]
        inv = ConstructiveNumber(Fraction(1, 1) / other.b, Fraction(1, 1) / other.a)
        return self.__mul__(inv)

    def __rtruediv__(self, other):
        return self._as_cn(other).__truediv__(self)

    # --- Сравнение (по середине интервала) ---
    def _mid(self):
        return (self.a + self.b) / 2

    def __eq__(self, other):
        other = self._as_cn(other)
        return self.a == other.a and self.b == other.b

    def __lt__(self, other):
        return self._mid() < self._as_cn(other)._mid()

    def __le__(self, other):
        return self._mid() <= self._as_cn(other)._mid()

    def __gt__(self, other):
        return self._mid() > self._as_cn(other)._mid()

    def __ge__(self, other):
        return self._mid() >= self._as_cn(other)._mid()

    # --- Получение действительного числа обратно ---
    def get_value(self, alpha: float = 0.5):
        """
        Возвращает точку внутри отрезка [a, b] по параметру alpha in [0, 1].
        alpha=0 -> a, alpha=1 -> b (линейная интерполяция).
        """
        if not (0 <= alpha <= 1):
            raise ValueError("alpha должен быть в диапазоне [0, 1]")
        return float(self.a + alpha * (self.b - self.a))
