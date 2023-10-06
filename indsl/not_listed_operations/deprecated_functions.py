# Copyright 2023 Cognite AS
import functools

from typing import List

from indsl import versioning
from indsl.data_quality import extreme
from indsl.detect import drift, ssid, vma
from indsl.filter import status_flag_filter, wavelet_filter
from indsl.not_listed_operations import no_op
from indsl.oil_and_gas import calculate_shutin_interval, productivity_index
from indsl.regression import poly_regression
from indsl.resample import interpolate, resample, resample_to_granularity
from indsl.smooth import alma, arma, butterworth, chebyshev, ewma, lwma, sg, sma
from indsl.statistics import remove_outliers
from indsl.ts_utils import (
    absolute,
    add,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctan2,
    arctanh,
    bin_map,
    ceil,
    clip,
    cos,
    cosh,
    deg2rad,
    differentiate,
    div,
    exp,
    floor,
    inv,
    log,
    log2,
    log10,
    logn,
    maximum,
    minimum,
    mod,
    mul,
    neg,
    power,
    rad2deg,
    round,
    sign,
    sin,
    sinh,
    sqrt,
    sub,
    tan,
    tanh,
    trapezoidal_integration,
)


old_to_new = {
    "SG_SMOOTHER": sg,
    "ALMA_SMOOTHER": alma,
    "ARMA_SMOOTHER": arma,
    "BTR_SMOOTHER": butterworth,
    "CHB_SMOOTHER": chebyshev,
    "EXP_WMA": ewma,
    "LINEAR_WMA": lwma,
    "SIMPLE_MA": sma,
    "INTERPOLATE": interpolate,
    "RESAMPLE_EXTENDED": resample,
    "RESAMPLE": resample_to_granularity,
    "POLY_REGRESSOR": poly_regression,
    "SHUTIN_CALC": calculate_shutin_interval,
    "PI_CALC": productivity_index,
    "PASSTHROUGH": no_op,
    "OUTLIER_DETECTOR": extreme,
    "DRIFT_DETECTOR": drift,
    "SS_DETECTOR": ssid,
    "VARIABLE_MA": vma,
    "WAVELET_FILTER": wavelet_filter,
    "STATUS_FLAG_FILTER": status_flag_filter,
    "OUTLIERS_REMOVE": remove_outliers,
    "ROUND": round,
    "FLOOR": floor,
    "CEIL": ceil,
    "SIGN": sign,
    "CLIP": clip,
    "MAX": maximum,
    "MIN": minimum,
    "BIN_MAP": bin_map,
    "SIN": sin,
    "COS": cos,
    "TAN": tan,
    "ARCSIN": arcsin,
    "ARCCOS": arccos,
    "ARCTAN": arctan,
    "ARCTAN2": arctan2,
    "DEG2RAD": deg2rad,
    "RAD2DEG": rad2deg,
    "SINH": sinh,
    "COSH": cosh,
    "TANH": tanh,
    "ARCSINH": arcsinh,
    "ARCCOSH": arccosh,
    "ARCTANH": arctanh,
    "ADD": add,
    "SUB": sub,
    "MUL": mul,
    "DIV": div,
    "POW": power,
    "INV": inv,
    "SQRT": sqrt,
    "NEG": neg,
    "ABS": absolute,
    "MOD": mod,
    "INTEGRATE": trapezoidal_integration,
    "DDX": differentiate,
    "EXP": exp,
    "LOG": log,
    "LOG2": log2,
    "LOG10": log10,
    "LOGN": logn,
}

for old_op_code, f in old_to_new.items():
    # If multiple versions exist, use version 1.0
    try:
        f = versioning.get(versioning.get_name(f), "1.0")
    except ValueError:
        pass

    # inDSLs versioning does not support giving a function two names
    # (here e.g. SIN and sin). We work around this by creating a
    # dummy-wrapper function
    def make_cpy(f):
        """Dummy wrapper function."""
        return functools.wraps(f)(lambda *args, **kwargs: f(*args, **kwargs))

    f_cpy = make_cpy(f)

    # Register function as a deprecated function with the old op code
    versioning.register(
        "1.0", old_op_code, deprecated=True, changelog="Deprecated function. Re-create node to upgrade"
    )(f_cpy)

    # Attach function to this module
    vars()[old_op_code] = f_cpy

# Expose only old style functions as module members
__all__: List[str] = list(old_to_new.keys())
