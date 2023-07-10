from __future__ import annotations

import cProfile
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
RT = TypeVar("RT")


def profile(filepath: str | Path) -> Callable:
    """Decorator that profiles a function and saves the .prof file generated.

    Arguments:
        filepath: Path to save code profile to

    Usage:
        ```
        @profile("path/to/file.prof")
        def func(*args, **kwargs):
            ...
        ```
    """

    def inner(func: Callable[P, RT]) -> Callable[P, RT]:
        """Returns a modified version of func, with code profiling."""

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> RT:
            """Modified version of `func` with code profiling."""

            with cProfile.Profile() as prof:
                result: RT = func(*args, **kwargs)

            prof.dump_stats(filepath)

            return result

        return wrapper

    return inner