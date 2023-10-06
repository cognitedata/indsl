# Copyright 2023 Cognite AS
from .change_point_detector import cpd_ed_pelt
from .cusum import cusum
from .drift_detector import drift
from .oscillation_detector import oscillation_detector
from .steady_state import ssd_cpd, ssid, vma
from .unchanged_signal_detector import unchanged_signal_detector


TOOLBOX_NAME = "Detect"

__all__ = [
    "drift",
    "ssid",
    "vma",
    "ssd_cpd",
    "cpd_ed_pelt",
    "unchanged_signal_detector",
    "cusum",
    "oscillation_detector",
]

__cognite__ = [
    "drift",
    "ssid",
    "vma",
    "ssd_cpd",
    "cpd_ed_pelt",
    "unchanged_signal_detector",
    "cusum",
    "oscillation_detector",
]
