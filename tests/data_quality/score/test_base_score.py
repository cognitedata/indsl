# Copyright 2021 Cognite AS

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_array_equal

from indsl.data_quality.score import DataQualityScore
from indsl.data_quality.score.base import DataQualityScoreAnalyser
from indsl.exceptions import UserTypeError, UserValueError
from indsl.ts_utils.utility_functions import create_series_from_timesteps
from indsl.type_check import check_types


# to test a class containing abstract methods we need
# to add a dummy implementation of the parent abstract methods
class DummyDataQualityScoreAnalyser(DataQualityScoreAnalyser):
    @check_types
    def __init__(self, series: pd.Series):
        """Gap based data quality scores.

        Args:
            series: Series to be analysed

        raises:
            UserValueError: If the series has less than 2 values
            UserValueError: If series has no time index
        """
        super().__init__(series)

    def compute_score(self, analysis_start: pd.Timestamp, analysis_end: pd.Timestamp) -> DataQualityScore:
        super().compute_score(analysis_start, analysis_end)
        return DataQualityScore(pd.Timestamp(0), pd.Timestamp(1), [])


@pytest.mark.core
def test_data_quality_score_init_errors():
    with pytest.raises(UserValueError) as excinfo:
        DataQualityScore(
            pd.Timestamp("2020-02-19"),
            pd.Timestamp("2020-02-20"),
            events=[(pd.Timestamp("2020-02-10"), pd.Timestamp("2020-02-11"))],
        )
    assert (
        "Expected event to be in analysis window, got event='2020-02-10 00:00:00-2020-02-11 00:00:00' and analysis_window='2020-02-19 00:00:00-2020-02-20 00:00:00'"
        == str(excinfo.value)
    )

    with pytest.raises(UserValueError) as excinfo:
        DataQualityScore(
            pd.Timestamp("2020-02-19"),
            pd.Timestamp("2020-02-24"),
            events=[(pd.Timestamp("2020-02-21"), pd.Timestamp("2020-02-20"))],
        )
    assert (
        "Expected start date of event to be before end date, got event_start='2020-02-21 00:00:00' and event_end='2020-02-20 00:00:00'"
        == str(excinfo.value)
    )


@pytest.mark.core
def test_data_quality_score_analyser_errors():
    series = pd.Series([1, 2], index=pd.to_datetime(["2020-02-01", "2020-02-03"]))
    analyser = DummyDataQualityScoreAnalyser(series)
    print(analyser)

    series = pd.Series([1, 2])
    with pytest.raises(UserTypeError) as excinfo:
        DummyDataQualityScoreAnalyser(series)
    assert "Expected a time series, got index type int64" == str(excinfo.value)

    with pytest.raises(UserValueError) as excinfo:
        analyser.compute_score(pd.Timestamp("2020-02-02"), pd.Timestamp("2020-02-01"))
    assert (
        "Expected analysis_start < analysis_end, got analysis_start '2020-02-02 00:00:00' and analysis_end '2020-02-01 00:00:00'"
        == str(excinfo.value)
    )


@pytest.mark.core
def test_data_quality_score_add_errors():
    score1 = DataQualityScore(pd.Timestamp("2020-02-02"), pd.Timestamp("2020-02-03"), events=[])
    score2 = DataQualityScore(pd.Timestamp("2020-02-04"), pd.Timestamp("2020-02-05"), events=[])

    with pytest.raises(UserValueError) as excinfo:
        score1 + score2
    assert (
        "Expected consecutive analysis periods in self and other, got self.analysis_end='2020-02-03 00:00:00' and other.analysis_start='2020-02-04 00:00:00'"
        == str(excinfo.value)
    )


@pytest.mark.core
def test_data_quality_score():
    score = DataQualityScore(
        pd.Timestamp("2020-02-09"),
        pd.Timestamp("2020-02-19"),
        events=[(pd.Timestamp("2020-02-10"), pd.Timestamp("2020-02-11"))],
    )

    assert score.analysis_start == pd.Timestamp("2020-02-09")
    assert score.analysis_end == pd.Timestamp("2020-02-19")
    np.testing.assert_array_equal(score.events, [np.asarray((pd.Timestamp("2020-02-10"), pd.Timestamp("2020-02-11")))])

    assert score.degradation == [0.1]
    assert score.score == 0.9


@pytest.mark.core
def test_data_quality_score_add():
    score1 = DataQualityScore(
        pd.Timestamp("2020-02-09"),
        pd.Timestamp("2020-02-19"),
        events=[(pd.Timestamp("2020-02-10"), pd.Timestamp("2020-02-11"))],
    )
    score2 = DataQualityScore(
        pd.Timestamp("2020-02-19"),
        pd.Timestamp("2020-02-29"),
        events=[(pd.Timestamp("2020-02-21"), pd.Timestamp("2020-02-22"))],
    )
    score = score1 + score2

    assert score.analysis_start == pd.Timestamp("2020-02-09")
    assert score.analysis_end == pd.Timestamp("2020-02-29")
    np.testing.assert_array_equal(
        score.events,
        [
            np.asarray((pd.Timestamp("2020-02-10"), pd.Timestamp("2020-02-11"))),
            np.asarray((pd.Timestamp("2020-02-21"), pd.Timestamp("2020-02-22"))),
        ],
    )
    assert score.degradation == [0.05, 0.05]
    assert score.score == 0.9


@pytest.mark.core
def test_data_quality_score_add_merged_events():
    score1 = DataQualityScore(
        pd.Timestamp("2020-02-09"),
        pd.Timestamp("2020-02-19"),
        events=[(pd.Timestamp("2020-02-18"), pd.Timestamp("2020-02-19"))],
    )
    score2 = DataQualityScore(
        pd.Timestamp("2020-02-19"),
        pd.Timestamp("2020-02-29"),
        events=[(pd.Timestamp("2020-02-19"), pd.Timestamp("2020-02-20"))],
    )
    score = score1 + score2

    assert score.analysis_start == pd.Timestamp("2020-02-09")
    assert score.analysis_end == pd.Timestamp("2020-02-29")
    np.testing.assert_array_equal(score.events, [(pd.Timestamp("2020-02-18"), pd.Timestamp("2020-02-20"))])
    assert score.degradation == [0.1]
    assert score.score == 0.9


@pytest.mark.parametrize(
    "gaps_series, expected_result",
    [
        (pd.Series([0, 0], index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"])), []),
        (pd.Series([1, 1], index=pd.DatetimeIndex(["2020-01-01", "2020-01-02"])), [["2020-01-01", "2020-01-02"]]),
        (
            pd.Series(
                [1, 1, 0, 0],
                index=pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-02 00:00:00.001", "2020-01-03"]),
            ),
            [["2020-01-01", "2020-01-02"]],
        ),
        (
            pd.Series(
                [0, 0, 1, 1],
                index=pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-02 00:00:00.001", "2020-01-03"]),
            ),
            [["2020-01-02 00:00:00.001", "2020-01-03"]],
        ),
        (
            pd.Series(
                [0, 0, 1, 1, 0, 0],
                index=pd.DatetimeIndex(
                    [
                        "2020-01-01",
                        "2020-01-02",
                        "2020-01-02 00:00:00.001000",
                        "2020-01-03",
                        "2020-01-03 00:00:00.001000",
                        "2020-01-04",
                    ]
                ),
            ),
            [["2020-01-02 00:00:00.001000", "2020-01-03"]],
        ),
        (
            pd.Series(
                [0, 0, 1, 1],
                index=pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-02 00:00:00.001000", "2020-01-04"]),
            ),
            [["2020-01-02 00:00:00.001000", "2020-01-04"]],
        ),
        # test the case where consecutive gaps are handled as different events (not merged)
        (
            pd.Series(
                [0, 0, 1, 1, 0, 0],
                index=pd.DatetimeIndex(
                    [
                        "2021-01-01 00:00:00.000",
                        "2021-01-01 00:09:59.999",
                        "2021-01-01 00:10:00.000",
                        "2021-01-01 00:19:59.999",
                        "2021-01-01 00:20:00.000",
                        "2021-01-01 00:30:00.000",
                    ]
                ),
            ),
            [("2021-01-01 00:10:00", "2021-01-01 00:15:00"), ("2021-01-01 00:15:00", "2021-01-01 00:19:59.999000")],
        ),
    ],
)
def test_convert_gaps_series_to_events(gaps_series, expected_result):
    # generate input data with a frequency of 1 minute and two consecutive gaps of 5 minutes
    minute = timedelta(minutes=1)
    timesteps = 10 * [minute] + [minute * 5] + [minute * 5] + 10 * [minute]
    series = create_series_from_timesteps(timesteps)
    analyser = DummyDataQualityScoreAnalyser(series)

    gap_events = analyser._convert_series_to_events(gaps_series)
    expected_events = np.asarray([[pd.Timestamp(t1), pd.Timestamp(t2)] for t1, t2 in expected_result])

    assert_array_equal(gap_events, expected_events)
