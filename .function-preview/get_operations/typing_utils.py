import enum
import typing

import pandas as pd


SUPPORTED_PARAM_TYPES = (
    str,
    float,
    int,
    bool,
    enum.Enum,
    pd.Timestamp,
    pd.Timedelta,
    typing.Optional,
    typing.Literal,
    typing.List,
)
BASIC_SUPPORTED_PARAM_TYPES = (str, float, int, bool)
PARAM_TYPES_FOR_CONVERSION = (enum.Enum, pd.Timestamp, pd.Timedelta)


def unwrap_optional_type_safe(any_type):
    """Unwrap an optional type if possible, else return original type"""
    try:
        return _unwrap_optional_type(any_type)
    except TypeError:
        return any_type


def _unwrap_optional_type(optional_type):
    """Return X for type Optional[X]"""
    # Optional types are represented as Union[X, NoneType]
    try:
        inner_type, none_type = typing.get_args(optional_type)
        assert isinstance(None, none_type)
    except Exception:
        raise TypeError(f"Expected Optional type of form Union[any, NoneType], got {optional_type}")
    return inner_type


def to_json_type(target_type, prefix=""):
    """Converts a Python type to a JSON compatible type"""

    origin_type = typing.get_origin(target_type)

    # Unwrap Optional types
    # Optional types are represented as Union[X, NoneType]
    if origin_type == typing.Union:
        inner_type = _unwrap_optional_type(target_type)
        return prefix + to_json_type(inner_type)

    # Unwrap List types
    elif origin_type == list:
        inner_type = typing.get_args(target_type)[0]
        return prefix + to_json_type(inner_type, prefix="array_")

    # Handle all other types
    if origin_type == typing.Literal:
        target_type = str

    elif issubclass(target_type, enum.Enum):
        target_type = str

    elif target_type not in SUPPORTED_PARAM_TYPES:
        raise ValueError(f"Expected type to be one of {SUPPORTED_PARAM_TYPES}, got {target_type}")

    return prefix + target_type.__name__.lower()
