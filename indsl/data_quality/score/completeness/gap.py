# Copyright 2023 Cognite AS
from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import pandas as pd

from indsl.data_quality.gaps_identification import (
    gaps_identification_iqr,
    gaps_identification_modified_z_scores,
    gaps_identification_threshold,
    gaps_identification_z_scores,
)
from indsl.data_quality.score.base import DataQualityScore, DataQualityScoreAnalyser
from indsl.type_check import check_types


class GapEventsList(list):
    """List of gap events with additional properties.

    Attributes:
        events: List of gap events.
    """

    def __init__(self, events: List[Tuple[pd.Timestamp, pd.Timestamp]]):
        """Initialize list of gap events."""
        super().__init__(events)

    @property
    def gaps_lengths(self):
        """Duration of gap events."""
        return [event[1] - event[0] for event in self]

    @property
    def _gaps_in_seconds(self):
        return [gap_length.total_seconds() for gap_length in self.gaps_lengths]

    @property
    def gaps_max(self):
        """Maximum gap event duration."""
        return max(self.gaps_lengths) if self.gaps_lengths else None

    @property
    def gaps_avg(self):
        """Average gap event duration."""
        try:
            timedeltas_in_seconds = sum(self._gaps_in_seconds)
            avg_in_seconds = timedeltas_in_seconds / len(self)
        except ZeroDivisionError:
            return None
        return pd.Timedelta(seconds=avg_in_seconds)


class GapDataQualityScoreAnalyser(DataQualityScoreAnalyser):
    """Gap based data quality scores."""

    @check_types
    def __init__(self, series: pd.Series):
        """Gap based data quality scores init function.

        Args:
            series: Series to be analysed

        Raises:
            UserValueError: If the series has less than 2 values
            UserValueError: If series has no time index
        """
        super().__init__(series)
        self._gap_detection_methods = {
            "iqr": gaps_identification_iqr,
            "z_scores": gaps_identification_z_scores,
            "modified_z_scores": gaps_identification_modified_z_scores,
            "threshold": gaps_identification_threshold,
        }

    @check_types
    def get_gaps(
        self,
        analysis_start: pd.Timestamp,
        analysis_end: pd.Timestamp,
        gap_detection_method: Literal["iqr", "z_scores", "modified_z_scores", "threshold"] = "iqr",
        **gap_detection_options: Optional[Union[pd.Timedelta, int, bool]],
    ) -> GapEventsList:
        """Calculate gap events.

        Args:
            analysis_start: Analyis start time
            analysis_end: Analyis end time
            gap_detection_method: Gap detection method
                Must be one of "iqr", "z_scores", "modified_z_scores"
            gap_detection_options: Arguments to gap detection method
                Provided as a keyword dictionary

        Returns:
            GapEventList: A GapEventList object
        """
        method = self._gap_detection_methods[gap_detection_method]
        gaps = method(self.series, **gap_detection_options)

        events = self._convert_series_to_events(gaps)
        events = self._filter_events_outside_analysis_period(events, analysis_start, analysis_end)
        # The first and last gap might range outside the analysis period. Let's fix this...
        events = self._limit_first_and_last_events_to_analysis_period(events, analysis_start, analysis_end)

        return GapEventsList(events)

    @check_types
    def compute_score(
        self,
        analysis_start: pd.Timestamp,
        analysis_end: pd.Timestamp,
        gap_detection_method: Literal["iqr", "z_scores", "modified_z_scores", "threshold"] = "iqr",
        **gap_detection_options: Optional[Union[pd.Timedelta, int, bool]],
    ) -> DataQualityScore:
        """Compute the gap analysis score.

        Args:
            analysis_start: Analyis start time
            analysis_end: Analyis end time
            gap_detection_method: Gap detection method
                Must be one of "iqr", "z_scores", "modified_z_scores"
            gap_detection_options: Arguments to gap detection method
                Provided as a keyword dictionary

        Returns:
            DataQualityScore: A GapDataQualityScore object

        Raises:
            UserValueError: If analysis_start < analysis_end
            UserValueError: If the analysis start and end timestamps are outside the range of the series index
        """
        super().compute_score(analysis_start, analysis_end)
        # Treat empty series as one gap
        if len(self.series) == 0:
            return DataQualityScore(analysis_start, analysis_end, [(analysis_start, analysis_end)])

        events = self.get_gaps(analysis_start, analysis_end, gap_detection_method, **gap_detection_options)

        return DataQualityScore(analysis_start, analysis_end, [tuple(event) for event in events])
