# Copyright 2023 Cognite AS
from .alma import alma
from .arma_smoother import arma
from .butterworth import butterworth
from .chebyshev import chebyshev
from .eweight_ma import ewma
from .lweight_ma import lwma
from .savitzky_golay import sg
from .simple_ma import sma


TOOLBOX_NAME = "Smooth"

__all__ = [
    "sg",
    "alma",
    "arma",
    "butterworth",
    "chebyshev",
    "ewma",
    "lwma",
    "sma",
]

__cognite__ = [
    "sg",
    "alma",
    "arma",
    "butterworth",
    "chebyshev",
    "ewma",
    "lwma",
    "sma",
]

# TODO: groupd smoothers into types: window and frequency
