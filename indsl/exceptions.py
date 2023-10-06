# Copyright 2023 Cognite AS
class UserException(Exception):
    """Exception raised when the error is due to the user."""

    pass


class UserValueError(ValueError, UserException):
    """Exception raised when the user provides a wrong value."""

    pass


class UserTypeError(TypeError, UserException):
    """Exception raised when the user provides a value with the wrong type."""

    pass


class UserRuntimeError(RuntimeError, UserException):
    """Exception raised when the error in runtime is due to the user."""

    pass


def _indsl_extras(lib: str, extra) -> str:
    return (
        f"This function requires {lib}. You might want to install `indsl[{extra}]` or `indsl[all]` instead of `indsl`."
    )


MATPLOTLIB_REQUIRED = _indsl_extras("matplotlib", "plot")
NUMBA_REQUIRED = _indsl_extras("numba", "numba")
FLUIDS_REQUIRED = _indsl_extras("fluids", "fluids")
CSAPS_REQUIRED = _indsl_extras("csaps", "modeling")
KNEED_REQUIRED = _indsl_extras("kneed", "modeling")
STATSMODELS_REQUIRED = _indsl_extras("statsmodels", "stats")
SCIKIT_IMAGE_REQUIRED = _indsl_extras("scikit-image", "scikit")
SCIKIT_LEARN_REQUIRED = _indsl_extras("scikit-learn", "scikit")
