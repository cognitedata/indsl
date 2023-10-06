# Copyright 2023 Cognite AS
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index


class DataQualityScore:
    """Data class storing the result of an event data quality analysis.

    Attributes:
        analysis_start: Analysis start time
        analysis_end: Analysis end time
        events: List of event events
            Represented as pairs of timestamps
    """

    @check_types
    def __init__(
        self,
        analysis_start: pd.Timestamp,
        analysis_end: pd.Timestamp,
        events: List[Tuple[pd.Timestamp, pd.Timestamp]],
    ):
        """Data quality score init function.

        Args:
            analysis_start: Analysis start time
            analysis_end: Analysis end time
            events: List of event events
                Represented as pairs of timestamps
        """
        for event_start, event_end in events:
            if event_start > event_end:
                raise UserValueError(
                    f"Expected start date of event to be before end date, got event_start='{event_start}' and event_end='{event_end}'"
                )
            if event_start < analysis_start or event_end > analysis_end:
                raise UserValueError(
                    f"Expected event to be in analysis window, got event='{event_start}-{event_end}' and analysis_window='{analysis_start}-{analysis_end}'"
                )

        self.analysis_start = analysis_start
        self.analysis_end = analysis_end
        self.events = events

    @property
    def degradation(self):
        """Degradation factors."""
        return [
            (event_end - event_start) / (self.analysis_end - self.analysis_start)
            for event_start, event_end in self.events
        ]

    @property
    def score(self) -> float:
        """Data quality score calculated as 1-sum(degradation)."""
        return 1.0 - sum(self.degradation)

    def __add__(self, other: DataQualityScore) -> DataQualityScore:
        """Return the union of two data quality results.

        Args:
            other: Other event data quality result

        Returns:
            DataQualityScore: The merged results

        Raises:
            UserValueError: If the two input results do not have a consequent analysis window
        """
        if self.analysis_end != other.analysis_start:
            raise UserValueError(
                f"Expected consecutive analysis periods in self and other, got self.analysis_end='{self.analysis_end}' and other.analysis_start='{other.analysis_start}'"
            )

        # Copy events to avoid side effects
        self_events = self.events.copy()
        other_events = other.events.copy()

        # Merge the last event of first score with the
        # first event of the second score if they are subsequent
        if len(self_events) > 0 and len(other_events) > 0 and self_events[-1][1] == other_events[0][0]:
            other_events[0] = (self_events.pop()[0], other_events[0][1])

        return DataQualityScore(self.analysis_start, other.analysis_end, self_events + other_events)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DataQualityScore):
            return (
                self.analysis_start == other.analysis_start
                and self.analysis_end == other.analysis_end
                and (np.asarray(self.events) == np.asarray(other.events)).all()
            )
        else:
            raise NotImplementedError(
                f"Equality comparison between type {type(self)} and {type(other)} not implemented"
            )


class DataQualityScoreAnalyser(ABC):
    """Object to calculate data quality scores.

    Attributes:
        series: time series
    """

    def __init__(self, series: pd.Series):
        """Data quality score analyser init function.

        Args:
            series: time series

        Raises:
            UserValueError: If series has no time index
        """
        validate_series_has_time_index(series)

        self.series = series

    @abstractmethod
    def compute_score(self, analysis_start: pd.Timestamp, analysis_end: pd.Timestamp) -> Optional[DataQualityScore]:
        """Compute data quality result.

        Args:
            analysis_start: analyis start time
            analysis_end: analyis end time

        Returns:
            DataQualityScore: A DataQualityScore object

        Raises:
            UserValueError: If analysis_start < analysis_end
            UserValueError: If the analysis start and end timestamps are outside the range of the series index
        """
        if analysis_start > analysis_end:
            raise UserValueError(
                f"Expected analysis_start < analysis_end, got analysis_start '{analysis_start}' and analysis_end '{analysis_end}'"
            )

        if len(self.series) < 1:
            return None

        if analysis_start < self.series.index[0]:
            raise UserValueError(
                f"Expected analysis_start to be equal or after the first timestamp in series, got analysis_start={analysis_start} and series.index[0]={self.series.index[0]}"
            )
        if analysis_end > self.series.index[-1]:
            raise UserValueError(
                f"Expected analysis_end to be before or equal the last timestamp in series, got analysis_end={analysis_end} and series.index[-1]={self.series.index[-1]}"
            )
        return None

    def _convert_series_to_events(self, series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        # Each gap in the input series is represented as a consecutive (1, 1) pair.
        # Hence filtering the 1 values and re-arranging the associated index as pairs
        # yields a list of the (start, end) gap events.
        events_array = list(series[series == 1].index.values.reshape(-1, 2))

        # gaps are merged if the input series has consecutive gaps (see generate_step_series)
        # to split the gaps again we need to loop though the events and add the original timestamps:
        # todo: remove this code if generate_step_series handles consecutive gaps without merging
        timestamps = np.array(self.series.index)
        events_array_unmerged = []

        for start, end in events_array:
            merged_timestamps = timestamps[(timestamps > start) & (timestamps < end)]
            if merged_timestamps.size > 0:
                events_unmerged = np.concatenate(
                    (merged_timestamps, merged_timestamps, np.array([start, end])), axis=0
                )  # duplicate timestamps to create events
                events_unmerged_sorted = np.sort(events_unmerged)
                events_array_unmerged.extend(events_unmerged_sorted.reshape(-1, 2))
            else:
                events_array_unmerged.append([start, end])

        return [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in events_array_unmerged]

    @staticmethod
    def _filter_events_outside_analysis_period(
        events: List[Tuple[pd.Timestamp, pd.Timestamp]], analysis_start: pd.Timestamp, analysis_end: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        return [(start, end) for start, end in events if end > analysis_start and start < analysis_end]

    @staticmethod
    def _limit_first_and_last_events_to_analysis_period(
        gaps: List[Tuple[pd.Timestamp, pd.Timestamp]], analysis_start: pd.Timestamp, analysis_end: pd.Timestamp
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        if len(gaps) == 0:
            return gaps

        first_gap = gaps[0]
        gaps[0] = (max(first_gap[0], analysis_start), first_gap[1])

        last_gap = gaps[-1]
        gaps[-1] = (last_gap[0], min(last_gap[1], analysis_end))

        return gaps
