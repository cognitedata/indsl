# Copyright 2023 Cognite AS
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.drilling.doc import doc


@pytest.mark.core
def test_doc_normal():
    """Test DOC calculation with normal values."""
    # Define input time series
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    # Example values:
    # ROP: 10 m/h
    # RPM: 120 rev/min
    # Expected DOC: 10 / (120 * 60) = 10 / 7200 = 0.001389 m/rev
    
    rop = pd.Series([10.0] * 6, index=date_range)  # m/h
    rpm = pd.Series([120.0] * 6, index=date_range)  # rev/min
    
    # Calculate expected DOC
    expected_doc = 10.0 / (120.0 * 60.0)
    
    # Calculate result
    result = doc(rop, rpm)
    
    # Check result (allow small numerical differences)
    expected_series = pd.Series([expected_doc] * 6, index=date_range, name="doc")
    assert_series_equal(result, expected_series, rtol=1e-6, check_names=False)


@pytest.mark.core
def test_doc_varying_values():
    """Test DOC calculation with varying input values."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=4)
    
    rop = pd.Series([8.0, 12.0, 6.0, 10.0], index=date_range)
    rpm = pd.Series([100.0, 150.0, 80.0, 120.0], index=date_range)
    
    result = doc(rop, rpm)
    
    # Calculate expected values for each point
    expected_values = []
    for i in range(4):
        expected_values.append(rop.iloc[i] / (rpm.iloc[i] * 60.0))
    
    expected_series = pd.Series(expected_values, index=date_range, name="doc")
    assert_series_equal(result, expected_series, rtol=1e-6, check_names=False)


@pytest.mark.core
def test_doc_with_nan():
    """Test DOC calculation with NaN values in input."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    rop = pd.Series([10.0, np.nan, 10.0, 10.0, 10.0, 10.0], index=date_range)
    rpm = pd.Series([120.0, 120.0, np.nan, 120.0, 120.0, 120.0], index=date_range)
    
    result = doc(rop, rpm)
    
    # Result should contain NaN where any input is NaN
    assert result.isnull().iloc[1]  # Second value should be NaN (ROP is NaN)
    assert result.isnull().iloc[2]  # Third value should be NaN (RPM is NaN)
    
    # Other values should be valid
    assert not result.isnull().iloc[0]
    assert not result.isnull().iloc[3]
    assert not result.isnull().iloc[4]
    assert not result.isnull().iloc[5]


@pytest.mark.core
def test_doc_all_nan_rop():
    """Test DOC returns all NaN when ROP is all NaN."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    rop = pd.Series([np.nan] * 6, index=date_range)
    rpm = pd.Series([120.0] * 6, index=date_range)
    
    result = doc(rop, rpm)
    
    # Result should be all NaN
    assert result.isnull().all()


@pytest.mark.core
def test_doc_all_nan_rpm():
    """Test DOC returns all NaN when RPM is all NaN."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    rop = pd.Series([10.0] * 6, index=date_range)
    rpm = pd.Series([np.nan] * 6, index=date_range)
    
    result = doc(rop, rpm)
    
    # Result should be all NaN
    assert result.isnull().all()


@pytest.mark.core
def test_doc_zero_rpm():
    """Test DOC returns NaN when RPM is zero."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)

    rop = pd.Series([10.0] * 6, index=date_range)
    rpm = pd.Series([0.0] * 6, index=date_range)  # Zero RPM

    result = doc(rop, rpm)
    # Should return NaN for all values when RPM is zero (division by zero)
    assert result.isnull().all()
    assert result.name == "doc"


@pytest.mark.core
def test_doc_negative_rpm():
    """Test DOC returns NaN when RPM is negative."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)

    rop = pd.Series([10.0] * 6, index=date_range)
    rpm = pd.Series([-120.0] * 6, index=date_range)  # Negative RPM

    result = doc(rop, rpm)
    # Should return NaN for all values when RPM is negative (negative denominator)
    assert result.isnull().all()
    assert result.name == "doc"


@pytest.mark.core
def test_doc_mixed_zero_and_nan_rpm():
    """Test DOC handles mixed zero and NaN RPM values correctly."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=4)

    rop = pd.Series([10.0, 10.0, 10.0, 10.0], index=date_range)
    rpm = pd.Series([120.0, np.nan, 0.0, 150.0], index=date_range)  # Mix of valid, NaN, and zero

    result = doc(rop, rpm)
    # Should return NaN for zero RPM, NaN for NaN RPM, and valid values for positive RPM
    assert not np.isnan(result.iloc[0])  # Valid value for RPM=120
    assert np.isnan(result.iloc[1])  # NaN for NaN RPM
    assert np.isnan(result.iloc[2])  # NaN for zero RPM
    assert not np.isnan(result.iloc[3])  # Valid value for RPM=150
    assert result.name == "doc"


@pytest.mark.core
def test_doc_zero_rop():
    """Test DOC calculation when ROP is zero (should return zero, not error)."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    rop = pd.Series([0.0] * 6, index=date_range)  # Zero ROP
    rpm = pd.Series([120.0] * 6, index=date_range)
    
    result = doc(rop, rpm)
    
    # Result should be all zeros
    expected_series = pd.Series([0.0] * 6, index=date_range, name="doc")
    assert_series_equal(result, expected_series, rtol=1e-6, check_names=False)


@pytest.mark.core
def test_doc_very_small_rpm():
    """Test DOC handles very small RPM values correctly (should not cause overflow)."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=4)
    
    rop = pd.Series([10.0, 10.0, 10.0, 10.0], index=date_range)
    rpm = pd.Series([1e-10, 1e-8, 1e-6, 120.0], index=date_range)  # Very small but positive RPM values
    
    result = doc(rop, rpm)
    
    # Very small RPM values should produce very large DOC values, but not inf
    # The first three should be valid (very large) numbers, not NaN or inf
    assert not np.isnan(result.iloc[0]) and not np.isinf(result.iloc[0])
    assert not np.isnan(result.iloc[1]) and not np.isinf(result.iloc[1])
    assert not np.isnan(result.iloc[2]) and not np.isinf(result.iloc[2])
    assert not np.isnan(result.iloc[3])  # Normal value should be valid
    assert result.name == "doc"


@pytest.mark.core
def test_doc_mixed_zero_rop_and_zero_rpm():
    """Test DOC handles edge case when both ROP and RPM are zero."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=4)
    
    rop = pd.Series([0.0, 10.0, 0.0, 10.0], index=date_range)
    rpm = pd.Series([0.0, 0.0, 120.0, 120.0], index=date_range)
    
    result = doc(rop, rpm)
    
    # When RPM is zero, result should be NaN (division by zero)
    assert np.isnan(result.iloc[0])  # Both zero -> NaN due to zero RPM
    assert np.isnan(result.iloc[1])  # Zero RPM -> NaN
    assert not np.isnan(result.iloc[2])  # Zero ROP with valid RPM -> 0.0
    assert not np.isnan(result.iloc[3])  # Both valid -> valid result
    assert result.iloc[2] == 0.0
    assert result.name == "doc"