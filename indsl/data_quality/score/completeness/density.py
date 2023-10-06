# Copyright 2023 Cognite AS
from __future__ import annotations

from typing import Literal, Optional, Union

import pandas as pd

from indsl.data_quality.low_density_identification import (
    low_density_identification_iqr,
    low_density_identification_modified_z_scores,
    low_density_identification_threshold,
    low_density_identification_z_scores,
)
from indsl.data_quality.score.base import DataQualityScore, DataQualityScoreAnalyser

# from indsl.exceptions import UserValueError
from indsl.type_check import check_types


class DensityDataQualityScoreAnalyser(DataQualityScoreAnalyser):
    """Density based data quality scores."""

    @check_types
    def __init__(self, series: pd.Series):
        """Density data quality score analyser init function.

        Args:
            series: Series to be analysed

        Raises:
            UserValueError: If the series has less than 2 values
            UserValueError: If series has no time index
        """
        super().__init__(series)
        self._low_density_detection_methods = {
            "iqr": low_density_identification_iqr,
            "z_scores": low_density_identification_z_scores,
            "modified_z_scores": low_density_identification_modified_z_scores,
            "threshold": low_density_identification_threshold,
        }

    @check_types
    def _get_low_density(
        self,
        analysis_start: pd.Timestamp,
        analysis_end: pd.Timestamp,
        low_density_detection_method: Literal["iqr", "z_scores", "modified_z_scores", "threshold"] = "iqr",
        **low_density_detection_options: Optional[Union[pd.Timedelta, int, bool]],
    ):
        method = self._low_density_detection_methods[low_density_detection_method]
        low_density_results = method(self.series, **low_density_detection_options)

        events = self._convert_series_to_events(low_density_results)
        events = self._filter_events_outside_analysis_period(events, analysis_start, analysis_end)
        # The first and last detected low density periods might range outside the analysis period. Let's fix this...
        events = self._limit_first_and_last_events_to_analysis_period(events, analysis_start, analysis_end)

        return events

    @check_types
    def compute_score(
        self,
        analysis_start: pd.Timestamp,
        analysis_end: pd.Timestamp,
        low_density_detection_method: Literal["iqr", "z_scores", "modified_z_scores", "threshold"] = "iqr",
        **low_density_detection_options: Optional[Union[pd.Timedelta, int, bool]],
    ) -> DataQualityScore:
        """Compute the low density analysis score.

        Args:
            analysis_start: Analyis start time
            analysis_end: Analyis end time
            low_density_detection_method: Low density detection method
                Must be one of "iqr", "z_scores", "modified_z_scores", "threshold". Default to "iqr".
            low_density_detection_options: Arguments to low density detection method
                Provided as a keyword dictionary

        Returns:
            DataQualityScore: A DataQualityScore object

        Raises:
            UserValueError: If analysis_start < analysis_end
            UserValueError: If the analysis start and end timestamps are outside the range of the series index
        """
        # initialize
        super().compute_score(analysis_start, analysis_end)
        # Treat empty series as one low density period
        if len(self.series) == 0:
            return DataQualityScore(analysis_start, analysis_end, [(analysis_start, analysis_end)])

        events = self._get_low_density(
            analysis_start, analysis_end, low_density_detection_method, **low_density_detection_options
        )

        return DataQualityScore(analysis_start, analysis_end, [tuple(event) for event in events])
