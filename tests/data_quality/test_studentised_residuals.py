# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.data_quality import extreme
from indsl.exceptions import UserValueError
from tests.detect.test_utils import RNGContext


@pytest.mark.core
def test_split_timeseries_returns_seconds_from_zero():
    """_split_timeseries should return [0.0, 1.0, 2.0, ...] for a 1-second DatetimeIndex (pandas 3 compat)."""
    from indsl.data_quality.outliers import _split_timeseries_into_time_and_value_arrays

    index = pd.date_range("2024-01-01", periods=3, freq="1s")
    data = pd.Series([1.0, 2.0, 3.0], index=index)
    x, y = _split_timeseries_into_time_and_value_arrays(data)
    np.testing.assert_array_almost_equal(x, [0.0, 1.0, 2.0])


@pytest.mark.core
def test_extreme_does_not_raise_for_modern_timestamps():
    """extreme() must not raise for a DatetimeIndex starting in 2024 (pandas 3 regression guard)."""
    with RNGContext():
        sig = np.random.normal(loc=100, size=100, scale=1)
    index = pd.date_range(start="2024-01-01", periods=len(sig), freq="1min")
    data = pd.Series(sig, index=index)
    result = extreme(data)
    assert isinstance(result, pd.Series)


@pytest.mark.core
def test_extreme_does_not_raise_linalg_error_for_duplicated_timestamps():
    """extreme() must not propagate a raw LinAlgError when timestamps are duplicated."""
    with RNGContext():
        sig = np.random.normal(loc=100, size=50, scale=1)
    index = pd.date_range(start="2024-01-01", periods=len(sig), freq="1min")
    # Duplicate every timestamp so the hat-matrix would be rank-deficient with inv()
    dup_index = index.repeat(2)
    dup_sig = np.repeat(sig, 2)
    data = pd.Series(dup_sig, index=dup_index)
    result = extreme(data)
    assert isinstance(result, pd.Series)


@pytest.mark.core
def test_anomaly_detector():
    # Create fake data
    with RNGContext():
        sig = np.random.normal(loc=100, size=1000, scale=1)
    index = pd.date_range(start="1970", periods=len(sig), freq="1min")

    # Add anomalies
    anom_num = [61, 61, 126, 83, 133, 21, 126, 126, 187, 13, 86, 140, 44, 146, 89, 135, 52, 31, 43, 197]
    anom_ids = [886, 427, 304, 871, 970, 316, 962, 634, 901, 8, 773, 72, 358, 222, 552, 369, 715, 263, 839, 689]

    sig[anom_ids] = anom_num

    clean_sig = np.delete(sig, anom_ids)

    anomalous_data = pd.Series(sig, index=index)
    clean_data = pd.Series(clean_sig, index=np.delete(index, anom_ids))

    # Call the anomaly detector
    res = extreme(anomalous_data)

    # Check results (all anomalies should be removed)
    assert_series_equal(res, clean_data)

    with pytest.raises(
        UserValueError, match="The significance level must be a number higher than or equal to 0 and lower than 1"
    ):
        _ = extreme(anomalous_data, alpha=-1)
        _ = extreme(anomalous_data, alpha=2)
