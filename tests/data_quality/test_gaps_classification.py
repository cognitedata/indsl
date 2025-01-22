import pytest
import pandas as pd
import numpy as np
from indsl.data_quality.gaps_classification import gaps_classification
from indsl.exceptions import UserTypeError




def test_no_gaps_series():
    """Test with no gaps in the series."""
    date_rng = pd.date_range(start='2025-01-01', end='2025-01-02', freq='1min')
    time_series = pd.Series(np.random.randn(len(date_rng)), index=date_rng)

    result = gaps_classification(time_series)
    assert result.empty, "Expected no gaps but found some."

def test_single_gap_series():
    """Test with a single gap in the series."""
    date_rng = pd.date_range(start='2025-01-01', end='2025-01-02', freq='1min')
    time_series = pd.Series(np.random.randn(len(date_rng)), index=date_rng)

    # Introduce a gap
    time_series = time_series.drop(time_series.index[100:120])

    result = gaps_classification(time_series)
    assert len(result) == 1, f"Expected 1 gap but found {len(result)}."
    assert result.iloc[0]['classification'] in ["Typical", "Significant", "Abnormal", "Singularities", "Extreme"], \
        "Classification is not in the expected categories."

def test_multiple_gaps_series():
    """Test with multiple gaps of varying sizes in the series."""
    date_rng = pd.date_range(start='2025-01-01', end='2025-01-02', freq='1min')
    time_series = pd.Series(np.random.randn(len(date_rng)), index=date_rng)

    # Introduce multiple gaps
    time_series = time_series.drop(time_series.index[100:120])
    time_series = time_series.drop(time_series.index[300:360])
    time_series = time_series.drop(time_series.index[600:900])

    result = gaps_classification(time_series)
    assert len(result) == 3, f"Expected 3 gaps but found {len(result)}."
    assert all(classification in ["Typical", "Significant", "Abnormal", "Singularities", "Extreme"]
               for classification in result['classification']), "Unexpected classification found."

def test_outlier_gap_series():
    """Test with an extreme outlier gap in the series."""
    date_rng = pd.date_range(start='2025-01-01', end='2025-01-03', freq='1min')
    time_series = pd.Series(np.random.randn(len(date_rng)), index=date_rng)

    # Introduce an extreme gap
    print(time_series)
    time_series = time_series.drop(time_series.index[500:2500])

    result = gaps_classification(time_series)
    assert len(result) == 1, f"Expected 1 gap but found {len(result)}."
    assert result.iloc[0]['classification'] == "Extreme", "Extreme gap not classified correctly."

def test_incorrect_input_series():
    """Test with incorrect input types."""
    with pytest.raises(UserTypeError):
        gaps_classification([1, 2, 3])  # Passing a list instead of a Series

    with pytest.raises(UserTypeError):
        gaps_classification(None)  # Passing None as input

def test_empty_series():
    """Test with an empty Series."""
    empty_series = pd.Series(dtype=float)
    with pytest.raises(UserTypeError):
        gaps_classification(empty_series)

def test_irregular_sampling_series():
    """Test with irregular sampling in the series."""
    date_rng = pd.to_datetime([
        '2025-01-01 00:00:00', '2025-01-01 00:01:00', '2025-01-01 00:02:00', '2025-01-01 00:03:00',
        '2025-01-01 00:04:00', '2025-01-01 00:10:00', '2025-01-01 00:11:00', '2025-01-01 00:14:00', '2025-01-01 00:15:00'
    ])
    time_series = pd.Series(np.random.randn(len(date_rng)), index=date_rng)

    result = gaps_classification(time_series)
    print('!!!!!')
    print(result)
    assert len(result) == 2, f"Expected 2 gaps but found {len(result)}."
    assert all(classification in ["Significant", "Abnormal", "Singularities", "Extreme"]
               for classification in result['classification']), "Unexpected classification found."
