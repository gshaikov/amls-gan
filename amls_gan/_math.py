import operator
from functools import reduce
from typing import TypeVar

T = TypeVar("T", bound=float)


def prod(xs: tuple[T, ...]) -> T:
    return reduce(operator.mul, xs, 1)
