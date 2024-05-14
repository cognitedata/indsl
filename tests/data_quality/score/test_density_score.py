# Copyright 2021 Cognite AS
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_almost_equal, assert_array_equal

from indsl.data_quality.score import DensityDataQualityScoreAnalyser


@pytest.mark.core
@pytest.mark.parametrize(
    "mock_gaps, analysis_start, analysis_end, expected_events, expected_degradation, expected_score",
    [
        # analyse period spans full input series range, 2 gaps
        (
            [1, 1, 0, 0, 0, 0, 1, 1],
            "2020-02-01",
            "2020-02-08",
            [["2020-02-01", "2020-02-02"], ["2020-02-07", "2020-02-08"]],
            [pd.Timedelta(days=1) / pd.Timedelta(days=7), pd.Timedelta(days=1) / pd.Timedelta(days=7)],
            0.714,
        ),
        # analyse period spans full input series range, 1 gap
        (
            [0, 0, 0, 1, 1, 0, 0, 0],
            "2020-02-01",
            "2020-02-08",
            [["2020-02-04", "2020-02-05"]],
            [pd.Timedelta(days=1) / pd.Timedelta(days=7)],
            0.857,
        ),
        # analyse period spans exactly gap range
        ([0, 0, 0, 1, 1, 0, 0, 0], "2020-02-04", "2020-02-05", [["2020-02-04", "2020-02-05"]], [1], 0),
        # analysis period contains no data points
        (
            [0, 0, 0, 1, 1, 0, 0, 0],
            "2020-02-04T01:00:00",
            "2020-02-04T23:00:00",
            [["2020-02-04T01:00:00", "2020-02-04T23:00:00"]],
            [1],
            0,
        ),
        # analyse period spans partial gap range
        (
            [0, 0, 0, 1, 1, 0, 0, 0],
            "2020-02-01",
            "2020-02-04T12:00",
            [["2020-02-04", "2020-02-04T12:00"]],
            [pd.Timedelta(days=0.5) / pd.Timedelta(days=3.5)],
            0.857,
        ),
        # data contains no gaps
        ([0, 0, 0, 0, 0, 0, 0, 0], "2020-02-01", "2020-02-08", [], [], 1),
        # data contains only gaps
        ([1, 1], "2020-02-01", "2020-02-02", [["2020-02-01", "2020-02-02"]], [1], 0),
    ],
)
@pytest.mark.parametrize("low_density_detection_method", ["iqr", "z_scores", "modified_z_scores", "threshold"])
def test_density_data_quality_score_analyser(
    mock_gaps,
    analysis_start,
    analysis_end,
    expected_events,
    expected_degradation,
    expected_score,
    low_density_detection_method,
):
    # create mock data
    N = len(mock_gaps)
    index = pd.date_range(start="2020-02-01", periods=N, freq="d")
    data = pd.Series(np.zeros(N), index=index)
    mock_return_value = pd.Series(mock_gaps, index=index)

    # no need to test point_density_threshold function here
    with mock.patch(
        f"indsl.data_quality.score.completeness.density.low_density_identification_{low_density_detection_method}",
        return_value=mock_return_value,
    ) as mock_function:  # noqa: F841
        analyser = DensityDataQualityScoreAnalyser(data)
        score = analyser.compute_score(
            analysis_start=pd.Timestamp(analysis_start),
            analysis_end=pd.Timestamp(analysis_end),
            low_density_detection_method=low_density_detection_method,
        )

    # check results
    if len(expected_events) > 0:
        assert_array_equal(score.events, [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in expected_events])
    else:
        assert len(expected_events) == len(score.events) == 0

    assert score.analysis_start == pd.Timestamp(analysis_start)
    assert score.analysis_end == pd.Timestamp(analysis_end)
    assert score.degradation == expected_degradation
    assert_almost_equal(score.score, expected_score, decimal=3)
