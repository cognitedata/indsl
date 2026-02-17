# Copyright 2026 Cognite AS
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal

from indsl.drilling.state_stand import state_stand
from indsl.exceptions import UserValueError


@pytest.mark.core
def test_state_stand_basic():
    """Test basic state stand detection with valid inputs."""
    # Create time series in milliseconds since epoch
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000) + i * 1000 for i in range(10)]
    
    date_range = pd.date_range(start=start_time, periods=10, freq="1s")
    
    # Create input time series
    time = pd.Series(time_ms, index=range(len(time_ms)))
    hkld = pd.Series([100.0] * 10, index=date_range)  # Normal hookload
    dmd = pd.Series([100.0 + i * 0.1 for i in range(10)], index=date_range)  # Increasing hole depth
    dbit = pd.Series([100.0 + i * 0.1 for i in range(10)], index=date_range)  # Bit depth matches hole depth
    bpos = pd.Series([10.0 + i * 0.05 for i in range(10)], index=date_range)  # Block moving slowly
    rpm = pd.Series([120.0] * 10, index=date_range)  # Rotating
    tors = pd.Series([5.0] * 10, index=date_range)  # Torque present
    spp = pd.Series([500.0] * 10, index=date_range)  # Pumping
    flow = pd.Series([200.0] * 10, index=date_range)  # Flow present
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Check that result is a DataFrame with correct columns
    assert isinstance(result, pd.DataFrame)
    assert "bvel" in result.columns
    assert "state_stand" in result.columns
    assert "state_mode" in result.columns
    assert len(result) == 10
    
    # Check that state_stand contains valid integer codes (0-8)
    assert all(result["state_stand"].between(0, 8))
    assert result["state_stand"].dtype == np.int64
    
    # Check that state_mode contains valid integer codes (0-15)
    assert all(result["state_mode"].between(0, 15))
    assert result["state_mode"].dtype == np.int64


@pytest.mark.core
def test_state_stand_slips():
    """Test detection of slips state (very low hookload)."""
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000) + i * 1000 for i in range(5)]
    date_range = pd.date_range(start=start_time, periods=5, freq="1s")
    
    time = pd.Series(time_ms, index=range(len(time_ms)))
    hkld = pd.Series([10.0] * 5, index=date_range)  # Very low hookload (slips)
    dmd = pd.Series([100.0] * 5, index=date_range)
    dbit = pd.Series([100.0] * 5, index=date_range)
    bpos = pd.Series([10.0] * 5, index=date_range)
    rpm = pd.Series([0.0] * 5, index=date_range)
    tors = pd.Series([0.0] * 5, index=date_range)
    spp = pd.Series([0.0] * 5, index=date_range)
    flow = pd.Series([0.0] * 5, index=date_range)
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Should detect slips: state_stand = 6 (ST_CONNECTION), state_mode = 3 (MD_SLIPS)
    assert all(result["state_stand"] == 6)
    assert all(result["state_mode"] == 3)

@pytest.mark.core
def test_state_stand_drilling():
    """Test detection of drilling state (on-bottom, rotation, pumping)."""
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000) + i * 1000 for i in range(5)]
    date_range = pd.date_range(start=start_time, periods=5, freq="1s")
    
    time = pd.Series(time_ms, index=range(len(time_ms)))
    hkld = pd.Series([200.0] * 5, index=date_range)  # Normal hookload
    dmd = pd.Series([100.0] * 5, index=date_range)
    dbit = pd.Series([99.8] * 5, index=date_range)  # On-bottom (within 0.5m)
    bpos = pd.Series([10.0] * 5, index=date_range)  # Stationary
    rpm = pd.Series([120.0] * 5, index=date_range)  # Rotating
    tors = pd.Series([5.0] * 5, index=date_range)  # Torque
    spp = pd.Series([500.0] * 5, index=date_range)  # Pumping
    flow = pd.Series([200.0] * 5, index=date_range)  # Flow
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Should detect drilling: state_stand = 1 (ST_DRILLING), state_mode = 1 (MD_DRILLROTATE)
    assert all(result["state_stand"] == 1)
    assert all(result["state_mode"] == 1)


@pytest.mark.core
def test_state_stand_block_velocity():
    """Test block velocity calculation."""
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000) + i * 1000 for i in range(10)]
    date_range = pd.date_range(start=start_time, periods=10, freq="1s")
    
    time = pd.Series(time_ms, index=range(len(time_ms)))
    # Block position moving at constant rate: 0.1 m/s
    bpos = pd.Series([10.0 + i * 0.1 for i in range(10)], index=date_range)
    
    # Other inputs (minimal values)
    hkld = pd.Series([100.0] * 10, index=date_range)
    dmd = pd.Series([100.0] * 10, index=date_range)
    dbit = pd.Series([100.0] * 10, index=date_range)
    rpm = pd.Series([0.0] * 10, index=date_range)
    tors = pd.Series([0.0] * 10, index=date_range)
    spp = pd.Series([0.0] * 10, index=date_range)
    flow = pd.Series([0.0] * 10, index=date_range)
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Block velocity should be approximately 0.1 m/s (with some numerical error)
    # First and last values may differ due to boundary conditions in gradient
    middle_values = result["bvel"].iloc[1:-1]
    assert np.allclose(middle_values, 0.1, rtol=0.1)


@pytest.mark.core
def test_state_stand_tripping_out():
    """Test detection of tripping out state (block moving up, no rotation)."""
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000) + i * 1000 for i in range(5)]
    date_range = pd.date_range(start=start_time, periods=5, freq="1s")
    
    time = pd.Series(time_ms, index=range(len(time_ms)))
    hkld = pd.Series([200.0] * 5, index=date_range)
    dmd = pd.Series([100.0] * 5, index=date_range)
    dbit = pd.Series([50.0] * 5, index=date_range)  # Off-bottom
    bpos = pd.Series([10.0 - i * 0.1 for i in range(5)], index=date_range)  # Moving up
    rpm = pd.Series([0.0] * 5, index=date_range)  # No rotation
    tors = pd.Series([0.0] * 5, index=date_range)
    spp = pd.Series([0.0] * 5, index=date_range)
    flow = pd.Series([0.0] * 5, index=date_range)
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Should detect tripping out: state_stand = 5 (ST_POOH), state_mode = 11 (MD_TRIPOUT)
    assert all(result["state_stand"] == 5)
    assert all(result["state_mode"] == 11)


@pytest.mark.core
def test_state_stand_insufficient_data():
    """Test handling of insufficient data (should return NaN values)."""
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000)]
    date_range = pd.date_range(start=start_time, periods=1, freq="1s")
    
    time = pd.Series(time_ms, index=range(len(time_ms)))
    hkld = pd.Series([100.0], index=date_range)
    dmd = pd.Series([100.0], index=date_range)
    dbit = pd.Series([100.0], index=date_range)
    bpos = pd.Series([10.0], index=date_range)
    rpm = pd.Series([0.0], index=date_range)
    tors = pd.Series([0.0], index=date_range)
    spp = pd.Series([0.0], index=date_range)
    flow = pd.Series([0.0], index=date_range)
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Should return NaN for bvel and ST_ABSENT/MD_ABSENT for states
    assert len(result) == 1
    assert np.isnan(result["bvel"].iloc[0])
    assert result["state_stand"].iloc[0] == 0  # ST_ABSENT
    assert result["state_mode"].iloc[0] == 0  # MD_ABSENT


@pytest.mark.core
def test_state_stand_with_nan():
    """Test handling of NaN values in inputs."""
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000) + i * 1000 for i in range(5)]
    date_range = pd.date_range(start=start_time, periods=5, freq="1s")
    
    time = pd.Series(time_ms, index=range(len(time_ms)))
    hkld = pd.Series([100.0, np.nan, 100.0, 100.0, 100.0], index=date_range)
    dmd = pd.Series([100.0] * 5, index=date_range)
    dbit = pd.Series([100.0] * 5, index=date_range)
    bpos = pd.Series([10.0] * 5, index=date_range)
    rpm = pd.Series([0.0] * 5, index=date_range)
    tors = pd.Series([0.0] * 5, index=date_range)
    spp = pd.Series([0.0] * 5, index=date_range)
    flow = pd.Series([0.0] * 5, index=date_range)
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Should still return results with NaN values marked as ST_ABSENT/MD_ABSENT
    assert len(result) == 5
    assert "bvel" in result.columns
    assert "state_stand" in result.columns
    assert "state_mode" in result.columns
    
    # Second value should be ST_ABSENT (0) and MD_ABSENT (0) due to NaN
    assert result["state_stand"].iloc[1] == 0
    assert result["state_mode"].iloc[1] == 0


@pytest.mark.core
def test_state_stand_circulating():
    """Test detection of circulating state (pumping, no rotation, off-bottom, stationary)."""
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000) + i * 1000 for i in range(5)]
    date_range = pd.date_range(start=start_time, periods=5, freq="1s")
    
    time = pd.Series(time_ms, index=range(len(time_ms)))
    hkld = pd.Series([200.0] * 5, index=date_range)
    dmd = pd.Series([100.0] * 5, index=date_range)
    dbit = pd.Series([50.0] * 5, index=date_range)  # Off-bottom
    bpos = pd.Series([10.0] * 5, index=date_range)  # Stationary
    rpm = pd.Series([0.0] * 5, index=date_range)  # No rotation
    tors = pd.Series([0.0] * 5, index=date_range)
    spp = pd.Series([500.0] * 5, index=date_range)  # Pumping
    flow = pd.Series([200.0] * 5, index=date_range)  # Flow
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Should detect: state_mode = 13 (MD_PUMP) - Static with pumping but not rotation
    assert all(result["state_mode"] == 13)


@pytest.mark.core
def test_state_stand_slide_drilling():
    """Test detection of slide drilling (on-bottom, no rotation, pumping)."""
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000) + i * 1000 for i in range(5)]
    date_range = pd.date_range(start=start_time, periods=5, freq="1s")
    
    time = pd.Series(time_ms, index=range(len(time_ms)))
    hkld = pd.Series([200.0] * 5, index=date_range)
    dmd = pd.Series([100.0] * 5, index=date_range)
    dbit = pd.Series([99.8] * 5, index=date_range)  # On-bottom
    bpos = pd.Series([10.0] * 5, index=date_range)  # Stationary
    rpm = pd.Series([0.0] * 5, index=date_range)  # No rotation
    tors = pd.Series([0.0] * 5, index=date_range)
    spp = pd.Series([500.0] * 5, index=date_range)  # Pumping
    flow = pd.Series([200.0] * 5, index=date_range)  # Flow
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Should detect: state_stand = 1 (ST_DRILLING), state_mode = 2 (MD_DRILLSLIDE)
    assert all(result["state_stand"] == 1)
    assert all(result["state_mode"] == 2)


@pytest.mark.core
def test_state_stand_offbottom():
    """Test detection of offbottom state (off-bottom, rotation, pumping)."""
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000) + i * 1000 for i in range(5)]
    date_range = pd.date_range(start=start_time, periods=5, freq="1s")
    
    time = pd.Series(time_ms, index=range(len(time_ms)))
    hkld = pd.Series([200.0] * 5, index=date_range)
    dmd = pd.Series([100.0] * 5, index=date_range)
    dbit = pd.Series([50.0] * 5, index=date_range)  # Off-bottom
    bpos = pd.Series([10.0] * 5, index=date_range)  # Stationary
    rpm = pd.Series([120.0] * 5, index=date_range)  # Rotating
    tors = pd.Series([5.0] * 5, index=date_range)
    spp = pd.Series([500.0] * 5, index=date_range)  # Pumping
    flow = pd.Series([200.0] * 5, index=date_range)  # Flow
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Should detect: state_stand = 2 (ST_OFFBOT), state_mode = 12 (MD_PUMPROTATE)
    assert all(result["state_stand"] == 2)
    assert all(result["state_mode"] == 12)


@pytest.mark.core
def test_state_stand_tripping_in():
    """Test detection of tripping in state (block moving down, no rotation, no pumping)."""
    start_time = datetime(2000, 10, 1, 0, 0, 0)
    time_ms = [int(pd.Timestamp(start_time).timestamp() * 1000) + i * 1000 for i in range(5)]
    date_range = pd.date_range(start=start_time, periods=5, freq="1s")
    
    time = pd.Series(time_ms, index=range(len(time_ms)))
    hkld = pd.Series([200.0] * 5, index=date_range)
    dmd = pd.Series([100.0] * 5, index=date_range)
    dbit = pd.Series([50.0] * 5, index=date_range)  # Off-bottom
    bpos = pd.Series([10.0 + i * 0.1 for i in range(5)], index=date_range)  # Moving down
    rpm = pd.Series([0.0] * 5, index=date_range)  # No rotation
    tors = pd.Series([0.0] * 5, index=date_range)
    spp = pd.Series([0.0] * 5, index=date_range)  # No pumping
    flow = pd.Series([0.0] * 5, index=date_range)
    
    result = state_stand(time, hkld, dmd, dbit, bpos, rpm, tors, spp, flow)
    
    # Should detect: state_stand = 4 (ST_RIH), state_mode = 7 (MD_TRIP)
    assert all(result["state_stand"] == 4)
    assert all(result["state_mode"] == 7)
