# Copyright 2023 Cognite AS
import pandas as pd

from indsl.versioning import register


@register(version="1.0", deprecated=True)
def versioning_test_op(series: pd.Series) -> pd.Series:
    """Old versioning test.

    This old function is used only for testing purposes.

    Args:
        series: Dummy input

    Returns:
        pandas.Series: Dummy output
    """
    return series
