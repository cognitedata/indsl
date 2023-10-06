# Copyright 2023 Cognite AS
from .simple_filters import status_flag_filter
from .wavelet_filter import wavelet_filter


TOOLBOX_NAME = "Filter"

__all__ = ["wavelet_filter", "status_flag_filter"]

__cognite__ = ["wavelet_filter", "status_flag_filter"]
