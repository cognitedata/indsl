# Copyright 2023 Cognite AS
from typing import Literal, Union

import pandas as pd

from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def logical_check(
    value_1: Union[pd.Series, float],
    value_2: Union[pd.Series, float],
    value_true: Union[pd.Series, float],
    value_false: Union[pd.Series, float],
    operation: Literal[
        "Equality", "Inequality", "Greater than", "Greater or equal than", "Smaller than", "Smaller or equal than"
    ] = "Equality",
) -> Union[pd.Series, float]:
    """Logical Check.

    Perform a logical check between time series/constants and returns the assigned time series/constants when the
    condition holds `true` or `false`. The logical check is performed following the format: `Value 1` {operator} `Value 2`,
    where the operator can be *Equality* (`==`), *Inequality* (`!=`), *Greater than* (`>`), *Greater or equal than* (`>=`),
    *Smaller than* (`<`) or *Smaller or equal than* (`<=`).

    Args:
        value_1: Value 1 - time series/number.
        value_2: Value 2 - time series/number.
        value_true: True - time series/number.
        value_false: False - time series/number.
        operation: Logical operation.

    Returns:
        Union[pd.Series, Number]: Time series/number.
    """

    def check_function(
        a: Union[pd.Series, float, int], b: Union[pd.Series, float, int]
    ) -> Union[pd.Series, bool]:  # Select operation
        if operation == "Equality":
            return a == b
        elif operation == "Inequality":
            return a != b
        elif operation == "Greater than":
            return a > b
        elif operation == "Greater or equal than":
            return a >= b
        elif operation == "Smaller than":
            return a < b
        elif operation == "Smaller or equal than":
            return a <= b
        else:
            return None

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
