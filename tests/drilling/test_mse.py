# Copyright 2023 Cognite AS
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.drilling.mse import mse


@pytest.mark.core
def test_mse_normal():
    """Test MSE calculation with normal values."""
    # Define input time series
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    # Example values:
    # Torque: 1000 N·m
    # RPM: 120 rpm
    # WOB: 50000 N
    # ROP: 10 m/h = 0.00278 m/s
    # Bit Area: 0.01 m² (10 cm diameter bit)
    
    torque = pd.Series([1000.0] * 6, index=date_range)
    rpm = pd.Series([120.0] * 6, index=date_range)
    wob = pd.Series([50000.0] * 6, index=date_range)
    rop = pd.Series([10.0] * 6, index=date_range)  # m/h
    bit_area = pd.Series([0.01] * 6, index=date_range)  # m²
    
    # Calculate expected MSE
    # Angular velocity = 120 * 2*pi / 60 = 12.566 rad/s
    # ROP in m/s = 10 / 3600 = 0.00278 m/s
    # Numerator = 1000 * 12.566 + 50000 * 0.00278 = 12566 + 139 = 12705
    # Denominator = 0.01 * 0.00278 = 0.0000278
    # MSE = 12705 / 0.0000278 = 457,194,245 Pa ≈ 457.2 MPa
    
    angular_velocity = 120.0 * 2.0 * np.pi / 60.0
    rop_m_per_s = 10.0 / 3600.0
    numerator = 1000.0 * angular_velocity + 50000.0 * rop_m_per_s
    denominator = 0.01 * rop_m_per_s
    expected_mse = numerator / denominator
    
    # Calculate result
    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Check result (allow small numerical differences)
    expected_series = pd.Series([expected_mse] * 6, index=date_range, name="mse")
    assert_series_equal(result, expected_series, rtol=1e-6, check_names=False)


@pytest.mark.core
def test_mse_varying_values():
    """Test MSE calculation with varying input values."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=4)
    
    torque = pd.Series([1000.0, 1500.0, 800.0, 1200.0], index=date_range)
    rpm = pd.Series([100.0, 150.0, 80.0, 120.0], index=date_range)
    wob = pd.Series([40000.0, 60000.0, 30000.0, 50000.0], index=date_range)
    rop = pd.Series([8.0, 12.0, 6.0, 10.0], index=date_range)
    bit_area = pd.Series([0.01, 0.01, 0.01, 0.01], index=date_range)
    
    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Calculate expected values for each point
    expected_values = []
    for i in range(4):
        angular_velocity = rpm.iloc[i] * 2.0 * np.pi / 60.0
        rop_m_per_s = rop.iloc[i] / 3600.0
        numerator = torque.iloc[i] * angular_velocity + wob.iloc[i] * rop_m_per_s
        denominator = bit_area.iloc[i] * rop_m_per_s
        expected_values.append(numerator / denominator)
    
    expected_series = pd.Series(expected_values, index=date_range, name="mse")
    assert_series_equal(result, expected_series, rtol=1e-6, check_names=False)


@pytest.mark.core
def test_mse_with_nan():
    """Test MSE calculation with NaN values in input."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    torque = pd.Series([1000.0, np.nan, 1000.0, 1000.0, 1000.0, 1000.0], index=date_range)
    rpm = pd.Series([120.0, 120.0, np.nan, 120.0, 120.0, 120.0], index=date_range)
    wob = pd.Series([50000.0, 50000.0, 50000.0, np.nan, 50000.0, 50000.0], index=date_range)
    rop = pd.Series([10.0, 10.0, 10.0, 10.0, np.nan, 10.0], index=date_range)
    bit_area = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01, np.nan], index=date_range)
    
    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Result should contain NaN where any input is NaN
    assert result.isnull().iloc[1]  # Second value should be NaN
    assert result.isnull().iloc[2]  # Third value should be NaN
    assert result.isnull().iloc[3]  # Fourth value should be NaN
    assert result.isnull().iloc[4]  # Fifth value should be NaN
    assert result.isnull().iloc[5]  # Sixth value should be NaN
    
    # First value should be valid
    assert not result.isnull().iloc[0]


@pytest.mark.core
def test_mse_all_nan_torque():
    """Test MSE returns all NaN when torque is all NaN."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    torque = pd.Series([np.nan] * 6, index=date_range)
    rpm = pd.Series([120.0] * 6, index=date_range)
    wob = pd.Series([50000.0] * 6, index=date_range)
    rop = pd.Series([10.0] * 6, index=date_range)
    bit_area = pd.Series([0.01] * 6, index=date_range)
    
    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Result should be all NaN
    assert result.isnull().all()


@pytest.mark.core
def test_mse_all_nan_rpm():
    """Test MSE returns all NaN when RPM is all NaN."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    torque = pd.Series([1000.0] * 6, index=date_range)
    rpm = pd.Series([np.nan] * 6, index=date_range)
    wob = pd.Series([50000.0] * 6, index=date_range)
    rop = pd.Series([10.0] * 6, index=date_range)
    bit_area = pd.Series([0.01] * 6, index=date_range)
    
    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Result should be all NaN
    assert result.isnull().all()


@pytest.mark.core
def test_mse_all_nan_wob():
    """Test MSE returns all NaN when WOB is all NaN."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    torque = pd.Series([1000.0] * 6, index=date_range)
    rpm = pd.Series([120.0] * 6, index=date_range)
    wob = pd.Series([np.nan] * 6, index=date_range)
    rop = pd.Series([10.0] * 6, index=date_range)
    bit_area = pd.Series([0.01] * 6, index=date_range)
    
    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Result should be all NaN
    assert result.isnull().all()


@pytest.mark.core
def test_mse_all_nan_rop():
    """Test MSE returns all NaN when ROP is all NaN."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    torque = pd.Series([1000.0] * 6, index=date_range)
    rpm = pd.Series([120.0] * 6, index=date_range)
    wob = pd.Series([50000.0] * 6, index=date_range)
    rop = pd.Series([np.nan] * 6, index=date_range)
    bit_area = pd.Series([0.01] * 6, index=date_range)
    
    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Result should be all NaN
    assert result.isnull().all()


@pytest.mark.core
def test_mse_all_nan_bit_area():
    """Test MSE returns all NaN when bit area is all NaN."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)
    
    torque = pd.Series([1000.0] * 6, index=date_range)
    rpm = pd.Series([120.0] * 6, index=date_range)
    wob = pd.Series([50000.0] * 6, index=date_range)
    rop = pd.Series([10.0] * 6, index=date_range)
    bit_area = pd.Series([np.nan] * 6, index=date_range)
    
    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Result should be all NaN
    assert result.isnull().all()


@pytest.mark.core
def test_mse_zero_rop():
    """Test MSE returns NaN when ROP is zero."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)

    torque = pd.Series([1000.0] * 6, index=date_range)
    rpm = pd.Series([120.0] * 6, index=date_range)
    wob = pd.Series([50000.0] * 6, index=date_range)
    rop = pd.Series([0.0] * 6, index=date_range)  # Zero ROP
    bit_area = pd.Series([0.01] * 6, index=date_range)

    result = mse(torque, rpm, wob, rop, bit_area)
    # Should return NaN for all values when ROP is zero (division by zero)
    assert result.isnull().all()
    assert result.name == "mse"


@pytest.mark.core
def test_mse_zero_bit_area():
    """Test MSE returns NaN when bit area is zero."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)

    torque = pd.Series([1000.0] * 6, index=date_range)
    rpm = pd.Series([120.0] * 6, index=date_range)
    wob = pd.Series([50000.0] * 6, index=date_range)
    rop = pd.Series([10.0] * 6, index=date_range)
    bit_area = pd.Series([0.0] * 6, index=date_range)  # Zero bit area

    result = mse(torque, rpm, wob, rop, bit_area)
    # Should return NaN for all values when bit area is zero (division by zero)
    assert result.isnull().all()
    assert result.name == "mse"


@pytest.mark.core
def test_mse_negative_rop():
    """Test MSE returns NaN when ROP is negative."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)

    torque = pd.Series([1000.0] * 6, index=date_range)
    rpm = pd.Series([120.0] * 6, index=date_range)
    wob = pd.Series([50000.0] * 6, index=date_range)
    rop = pd.Series([-5.0] * 6, index=date_range)  # Negative ROP
    bit_area = pd.Series([0.01] * 6, index=date_range)

    result = mse(torque, rpm, wob, rop, bit_area)
    # Should return NaN for all values when ROP is negative (negative denominator)
    assert result.isnull().all()
    assert result.name == "mse"


@pytest.mark.core
def test_mse_negative_bit_area():
    """Test MSE returns NaN when bit area is negative."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=6)

    torque = pd.Series([1000.0] * 6, index=date_range)
    rpm = pd.Series([120.0] * 6, index=date_range)
    wob = pd.Series([50000.0] * 6, index=date_range)
    rop = pd.Series([10.0] * 6, index=date_range)
    bit_area = pd.Series([-0.01] * 6, index=date_range)  # Negative bit area

    result = mse(torque, rpm, wob, rop, bit_area)
    # Should return NaN for all values when bit area is negative (negative denominator)
    assert result.isnull().all()
    assert result.name == "mse"


@pytest.mark.core
def test_mse_very_small_rop():
    """Test MSE handles very small ROP values correctly (should not cause overflow)."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=4)

    torque = pd.Series([1000.0] * 4, index=date_range)
    rpm = pd.Series([120.0] * 4, index=date_range)
    wob = pd.Series([50000.0] * 4, index=date_range)
    rop = pd.Series([1e-10, 1e-8, 1e-6, 10.0], index=date_range)  # Very small but positive ROP values
    bit_area = pd.Series([0.01] * 4, index=date_range)

    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Very small ROP values should produce very large MSE values, but not inf
    # The first three should be valid (very large) numbers, not NaN or inf
    assert not np.isnan(result.iloc[0]) and not np.isinf(result.iloc[0])
    assert not np.isnan(result.iloc[1]) and not np.isinf(result.iloc[1])
    assert not np.isnan(result.iloc[2]) and not np.isinf(result.iloc[2])
    assert not np.isnan(result.iloc[3])  # Normal value should be valid
    assert result.name == "mse"


@pytest.mark.core
def test_mse_very_small_bit_area():
    """Test MSE handles very small bit area values correctly (should not cause overflow)."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=4)

    torque = pd.Series([1000.0] * 4, index=date_range)
    rpm = pd.Series([120.0] * 4, index=date_range)
    wob = pd.Series([50000.0] * 4, index=date_range)
    rop = pd.Series([10.0] * 4, index=date_range)
    bit_area = pd.Series([1e-10, 1e-8, 1e-6, 0.01], index=date_range)  # Very small but positive bit area values

    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Very small bit area values should produce very large MSE values, but not inf
    # The first three should be valid (very large) numbers, not NaN or inf
    assert not np.isnan(result.iloc[0]) and not np.isinf(result.iloc[0])
    assert not np.isnan(result.iloc[1]) and not np.isinf(result.iloc[1])
    assert not np.isnan(result.iloc[2]) and not np.isinf(result.iloc[2])
    assert not np.isnan(result.iloc[3])  # Normal value should be valid
    assert result.name == "mse"


@pytest.mark.core
def test_mse_very_small_denominator():
    """Test MSE handles very small denominator (bit_area * rop) correctly."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=4)

    torque = pd.Series([1000.0] * 4, index=date_range)
    rpm = pd.Series([120.0] * 4, index=date_range)
    wob = pd.Series([50000.0] * 4, index=date_range)
    # Create very small denominator: bit_area * rop_m_per_s
    rop = pd.Series([1e-5, 1e-3, 0.1, 10.0], index=date_range)  # Very small ROP
    bit_area = pd.Series([1e-5, 1e-3, 0.1, 0.01], index=date_range)  # Very small bit area

    result = mse(torque, rpm, wob, rop, bit_area)
    
    # Very small denominators should produce very large MSE values, but not inf
    # All should be valid numbers, not NaN or inf
    for i in range(4):
        assert not np.isnan(result.iloc[i]) and not np.isinf(result.iloc[i])
    assert result.name == "mse"


@pytest.mark.core
def test_mse_mixed_zero_rop_and_zero_bit_area():
    """Test MSE handles edge case when both ROP and bit area are zero."""
    date_range = pd.date_range(start=datetime(2000, 10, 1), end=datetime(2000, 10, 10), periods=4)

    torque = pd.Series([1000.0] * 4, index=date_range)
    rpm = pd.Series([120.0] * 4, index=date_range)
    wob = pd.Series([50000.0] * 4, index=date_range)
    rop = pd.Series([0.0, 10.0, 0.0, 10.0], index=date_range)
    bit_area = pd.Series([0.0, 0.0, 0.01, 0.01], index=date_range)

    result = mse(torque, rpm, wob, rop, bit_area)
    
    # When denominator is zero (ROP=0 or bit_area=0), result should be NaN
    assert np.isnan(result.iloc[0])  # Both zero -> NaN due to zero denominator
    assert np.isnan(result.iloc[1])  # Zero bit area -> NaN
    assert np.isnan(result.iloc[2])  # Zero ROP -> NaN
    assert not np.isnan(result.iloc[3])  # Both valid -> valid result
    assert result.name == "mse"