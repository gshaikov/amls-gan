from typing import TypeVar

T = TypeVar("T")


def not_none(x: T | None) -> T:
    """
    Pass-through if not None;
    explode otherwise.
    """
    if x is None:
        raise ValueError("None not allowed.")
    return x
