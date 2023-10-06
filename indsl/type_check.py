# Copyright 2023 Cognite AS
from functools import wraps

from typeguard import TypeCheckError, typechecked

from indsl.exceptions import UserTypeError


def error_handling(operation):
    """Decorator that catches TypeError and wraps to inDSL specific error."""

    @wraps(operation)
    def wrapper(*args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except TypeCheckError as e:
            raise UserTypeError(str(e)) from e

    return wrapper


def check_types(operation):
    """Decorator to check types of inputs and outputs of a function.

    Decorator uses typeguard library to validate arguments of a
    function, and then wraps a TypeError to UserTypeError which is
    specific for inDSL library
    """
    return error_handling(typechecked(operation))
