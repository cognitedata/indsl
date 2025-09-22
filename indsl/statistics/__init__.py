# Copyright 2023 Cognite AS
from .correlation import pearson_correlation
from .outliers import detect_outliers, remove_outliers


TOOLBOX_NAME = "Statistics"

__all__ = ["detect_outliers", "pearson_correlation", "remove_outliers"]

__cognite__ = ["detect_outliers", "remove_outliers", "pearson_correlation"]
