# Copyright 2023 - 2026 Cognite AS
from typing import Literal

import pandas as pd

from indsl import versioning
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@versioning.register(
    "2.0",
    changelog="Correct spelling of 'Greater or equal than' and 'Smaller or equal than'. Added support for None as true/false values.",
)
@check_types
def logical_check(
    value_1: pd.Series | float,
    value_2: pd.Series | float,
    value_true: pd.Series | float | None = 1,
    value_false: pd.Series | float | None = 0,
    operation: Literal[
        "Equality", "Inequality", "Greater than", "Greater than or equal to", "Smaller than", "Smaller than or equal to"
    ] = "Equality",
) -> pd.Series | float | None:
    """Logical Check.

    Perform a logical check between time series/constants and returns the assigned time series/constants when the
    condition holds `true` or `false`. The logical check is performed following the format: `Value 1` {operator} `Value 2`,
    where the operator can be *Equality* (`==`), *Inequality* (`!=`), *Greater than* (`>`), *Greater than or equal to* (`>=`),
    *Smaller than* (`<`) or *Smaller than or equal to* (`<=`).

    Args:
        value_1: Value 1 - time series/number.
        value_2: Value 2 - time series/number.
        value_true: What to return when true.
            If value_1 or value_2 are time series, and value_true is a constant, the constant will be returned at the indexes where the condition holds true.
            If both value_1 and value_2 are constants, and value_true is a time series, the time series will be returned if the condition holds true.
        value_false: What to return when false.
            If value_1 or value_2 are time series, and value_false is a constant, the constant will be returned at the indexes where the condition holds false.
            If both value_1 and value_2 are constants, and value_false is a time series, the time series will be returned if the condition holds false.
        operation: Logical operation.

    Returns:
        Time series, number or None.
    """

    def check_function(a: pd.Series | float | int, b: pd.Series | float | int) -> pd.Series | bool:  # Select operation
        if operation == "Equality":
            return a == b
        elif operation == "Inequality":
            return a != b
        elif operation == "Greater than":
            return a > b
        elif operation == "Greater than or equal to":
            return a >= b
        elif operation == "Smaller than":
            return a < b
        elif operation == "Smaller than or equal to":
            return a <= b

    # Case 1: both inputs are constants, we execute the logical check and return the associated value
    if isinstance(value_1, float) and isinstance(value_2, float):
        return value_true if check_function(value_1, value_2) else value_false

    # Case 2: one of the inputs is a time series

    # First we align all the time series
    value_1, value_2, value_false, value_true = auto_align([value_1, value_2, value_false, value_true])

    # We take the index of one of the input time series
    result = value_1.copy() if isinstance(value_1, pd.Series) else value_2.copy()
    # Rename the output series
    result.name = "result"
    # Set all values to the false state
    result.loc[:] = value_false
    # At the indexes where the logical check holds true, replace the values with the true state
    result.loc[check_function(value_1, value_2)] = value_true

    return result
