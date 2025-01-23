# Copyright 2023 Cognite AS
from typing import List

import numpy as np
import pandas as pd

from indsl.exceptions import SCIKIT_LEARN_REQUIRED
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index, validate_series_is_not_empty


@check_types
def gaps_classification(x: pd.Series, eps: float = 0.5, min_samples: int = 2, std_thresholds: List[int] = [1, 2, 3]):
    """Gaps Classification.

    Classify gaps in a time series dataset into categories based on duration and statistical properties.  
    DBSCAN is first used to determine data to be classified as Extreme.  
    Remaining data is classified as Typical, Significant, Abnormal, or Singularities depending on standard deviation thresholds.

    Args:
         x: Time series
         eps: The maximum distance between samples for clustering in DBSCAN. Defaults to 0.5.
         min_samples: The minimum number of samples in a cluster for DBSCAN. Defaults to 2.
         std_thresholds: Thresholds for classifying gaps based on standard deviations. Defaults to [1, 2, 3].

    Returns:
         pd.DataFrame: A DataFrame with gap start, gap end, duration, and classification.
    """
    validate_series_has_time_index(x)
    validate_series_is_not_empty(x)
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError(SCIKIT_LEARN_REQUIRED)
    # Calculate gaps
    x = x.sort_index()
    deltas = x.index.to_series().diff().dt.total_seconds()
    avg_sampling_rate = deltas.mean()
    std_sampling_rate = deltas.std()

    # Identify gaps
    gaps = deltas[deltas > avg_sampling_rate * 1.5]  # Threshold for detecting gaps
    gap_durations = gaps.values  # Gap durations in seconds

    # Classify using DBSCAN
    if len(gap_durations) > 0:
        dbscan = DBSCAN(eps=eps * std_sampling_rate, min_samples=min_samples)
        clusters = dbscan.fit_predict(gap_durations.reshape(-1, 1))
    else:
        clusters = np.array([])

    # Classification rules
    classifications = []
    for duration, cluster in zip(gap_durations, clusters):
        if cluster == -1:  # Outlier
            classifications.append("Extreme")
        else:
            deviation = (duration - avg_sampling_rate) / std_sampling_rate
            if deviation < std_thresholds[0]:
                classifications.append("Typical")
            elif deviation < std_thresholds[1]:
                classifications.append("Significant")
            elif deviation < std_thresholds[2]:
                classifications.append("Abnormal")
            else:
                classifications.append("Singularities")

    return pd.DataFrame(
        {
            "gap_start": gaps.index - pd.to_timedelta(gaps.values, unit="s"),
            "gap_end": gaps.index,
            "duration": gap_durations,
            "classification": classifications,
        }
    )
