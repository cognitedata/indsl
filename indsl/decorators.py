# Copyright 2023 Cognite AS
from typing import Any, Callable, Optional


try:
    from numba import jit
except ImportError:
    ...  # Workaround for typeguard

    def jit(fn: Optional[Callable[..., Any]] = None, **kwargs) -> Callable[..., Any]:  # noqa: D103
        # Check if this is a decorator with arguments or not
        if fn is None:
            # Decorator has arguments, ignore arguments and return a
            # function that accepts a function as argument
            return lambda f: f

        # No arguments, just return the provided function directly
        return fn


try:
    from numba import njit
except ImportError:
    ...  # Workaround for typeguard

    def njit(fn: Optional[Callable[..., Any]] = None, **kwargs) -> Callable[..., Any]:  # noqa: D103
        # Check if this is a decorator with arguments or not
        if fn is None:
            # Decorator has arguments, ignore arguments and return a
            # function that accepts a function as argument
            return lambda f: f

        # No arguments, just return the provided function directly
        return fn


__all__ = ["jit", "njit"]
