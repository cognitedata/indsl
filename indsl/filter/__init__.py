# Copyright 2023 Cognite AS
from .hilbert_huang_transform import hilbert_huang_transform
from .simple_filters import status_flag_filter
from .wavelet_filter import wavelet_filter


TOOLBOX_NAME = "Filter"

__all__ = ["hilbert_huang_transform", "wavelet_filter", "status_flag_filter"]

__cognite__ = ["wavelet_filter", "status_flag_filter"]
