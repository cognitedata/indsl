# Copyright 2021 Cognite AS

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal

from indsl.data_quality.score import GapDataQualityScoreAnalyser


@pytest.mark.core
@pytest.mark.parametrize(
    "mock_gaps, analysis_start, analysis_end, expected_events, expected_degradation, expected_score, expected_gap_detection_call_count",
    [
        # analyse period spans full input series range
        ([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], "2020-02-01", "2020-02-11", [["2020-02-04", "2020-02-05"]], [0.1], 0.9, 1),
        # analyse period spans exactly gap range
        ([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], "2020-02-04", "2020-02-05", [["2020-02-04", "2020-02-05"]], [1.0], 0.0, 1),
        # analysis period contains no data points
        (
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            "2020-02-04T01:00:00",
            "2020-02-04T23:00:00",
            [["2020-02-04T01:00:00", "2020-02-04T23:00:00"]],
            [1.0],
            0.0,
            1,
        ),
        # analyse period spans partial gap range
        (
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            "2020-02-01",
            "2020-02-04T12:00",
            [["2020-02-04", "2020-02-04T12:00"]],
            [0.5 / 3.5],
            1 - 0.5 / 3.5,
            1,
        ),
        # data contains no gaps
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "2020-02-01", "2020-02-04", [], [], 1, 1),
        # data contains only gaps
        ([1, 1], "2020-02-01", "2020-02-02", [["2020-02-01", "2020-02-02"]], [1], 0, 1),
    ],
)
@pytest.mark.parametrize("gap_detection_method", ["iqr", "z_scores", "modified_z_scores", "threshold"])
def test_gap_data_quality_score_analyser(
    gap_detection_method,
    mock_gaps,
    analysis_start,
    analysis_end,
    expected_events,
    expected_degradation,
    expected_score,
    expected_gap_detection_call_count,
):
    # create mock data
    N = len(mock_gaps)
    index = pd.date_range(start="2020-02-01", periods=N, freq="D")
    data = pd.Series(np.zeros(N), index=index)
    mock_return_value = pd.Series(mock_gaps, index=index)

    with mock.patch(
        f"indsl.data_quality.score.completeness.gap.gaps_identification_{gap_detection_method}",
        return_value=mock_return_value,
    ) as mock_function:
        analyser = GapDataQualityScoreAnalyser(data)
        score = analyser.compute_score(pd.Timestamp(analysis_start), pd.Timestamp(analysis_end), gap_detection_method)
    assert mock_function.call_count == expected_gap_detection_call_count

    assert score.analysis_start == pd.Timestamp(analysis_start)
    assert score.analysis_end == pd.Timestamp(analysis_end)
    if len(expected_events) > 0:
        assert_array_equal(score.events, [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in expected_events])
    else:
        assert len(expected_events) == len(score.events) == 0
    assert score.degradation == expected_degradation
    assert score.score == expected_score


@pytest.mark.core
@pytest.mark.parametrize(
    "mock_gaps, analysis_start, analysis_end, expected_gap_detection_call_count, gaps_lengths, gaps_in_seconds, gaps_max, gaps_avg",
    [
        # analyse period spans full input series range
        (
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            "2020-02-01",
            "2020-02-11",
            1,
            [pd.Timedelta("1 days 00:00:00")],
            [86400.0],
            pd.Timedelta("1 days 00:00:00"),
            pd.Timedelta("1 days 00:00:00"),
        ),
        # analyse period spans exactly gap range
        (
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            "2020-02-04",
            "2020-02-05",
            1,
            [pd.Timedelta("1 days 00:00:00")],
            [86400.0],
            pd.Timedelta("1 days 00:00:00"),
            pd.Timedelta("1 days 00:00:00"),
        ),
        # analysis period contains no data points
        (
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            "2020-02-04T01:00:00",
            "2020-02-04T23:00:00",
            1,
            [pd.Timedelta("0 days 22:00:00")],
            [79200.0],
            pd.Timedelta("0 days 22:00:00"),
            pd.Timedelta("0 days 22:00:00"),
        ),
        # analyse period spans partial gap range
        (
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            "2020-02-01",
            "2020-02-04T12:00",
            1,
            [pd.Timedelta("0 days 12:00:00")],
            [43200.0],
            pd.Timedelta("0 days 12:00:00"),
            pd.Timedelta("0 days 12:00:00"),
        ),
        # data contains no gaps
        ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "2020-02-01", "2020-02-04", 1, [], [], None, None),
        # data contains only gaps
        (
            [1, 1],
            "2020-02-01",
            "2020-02-02",
            1,
            [pd.Timedelta("1 days 00:00:00")],
            [86400.0],
            pd.Timedelta("1 days 00:00:00"),
            pd.Timedelta("1 days 00:00:00"),
        ),
    ],
)
@pytest.mark.parametrize("gap_detection_method", ["iqr", "z_scores", "modified_z_scores", "threshold"])
def test_get_gaps(
    gap_detection_method,
    mock_gaps,
    analysis_start,
    analysis_end,
    expected_gap_detection_call_count,
    gaps_lengths,
    gaps_in_seconds,
    gaps_max,
    gaps_avg,
):
    # create mock data
    N = len(mock_gaps)
    index = pd.date_range(start="2020-02-01", periods=N, freq="D")
    data = pd.Series(np.zeros(N), index=index)
    mock_return_value = pd.Series(mock_gaps, index=index)

    with mock.patch(
        f"indsl.data_quality.score.completeness.gap.gaps_identification_{gap_detection_method}",
        return_value=mock_return_value,
    ) as mock_function:
        analyser = GapDataQualityScoreAnalyser(data)
        gaps = analyser.get_gaps(pd.Timestamp(analysis_start), pd.Timestamp(analysis_end), gap_detection_method)
    assert mock_function.call_count == expected_gap_detection_call_count

    assert gaps.gaps_lengths == gaps_lengths
    assert gaps._gaps_in_seconds == gaps_in_seconds
    assert gaps.gaps_max == gaps_max
    assert gaps.gaps_avg == gaps_avg
