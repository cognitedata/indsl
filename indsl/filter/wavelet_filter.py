# Copyright 2023 Cognite AS

from typing import Literal

import pandas as pd

from indsl import versioning
from indsl.exceptions import SCIKIT_IMAGE_REQUIRED, UserValueError
from indsl.type_check import check_types

from . import wavelet_filter_v1  # noqa


wavelet_options = Literal[
    "db1",
    "db2",
    "db3",
    "db4",
    "db5",
    "db6",
    "db7",
    "db8",
    "sym2",
    "sym3",
    "sym4",
    "coif1",
    "coif2",
    "coif3",
    "coif4",
    "coif5",
]


@versioning.register(version="2.0", changelog="updated wavelet option type, added coif[1-5] as wavelet option")
@check_types
def wavelet_filter(data: pd.Series, level: int = 2, wavelet: wavelet_options = "db8") -> pd.Series:
    """Wavelet de-noising.

    Filtering industrial data using wavelets can be very powerful as it uses a *dual* frequency-time
    representation of the original signal, which allows separating noise frequencies from valuable signal frequencies.
    For more on wavelet filter or other application, see https://en.wikipedia.org/wiki/Wavelet

    Args:
        data: Time series.
            The data to be filtered. The series must have a pandas.DatetimeIndex.
        level: Level.
            The number of wavelet decomposition levels (typically 1 through 6) to use.
        wavelet: Type.
            The default is a Daubechies wavelet of order 8 (*db8*). For other types of wavelets, see the
            `pywavelets package <https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html>`_.
            The thresholding methods assume an orthogonal wavelet transform and may not choose the threshold
            appropriately for biorthogonal wavelets. Orthogonal wavelets are desirable because white noise in
            the input remains white noise in the sub-bands. Therefore one should choose one of the db[1-20], sym[2-20],
            or coif[1-5] type wavelet filters.

    Raises:
        UserValueError: The level value needs to be a positive integer
        UserValueError: The level value can not exceed the length of data points

    Returns:
        pandas.Series: Filtered time series.
    """
    try:
        from skimage.restoration import denoise_wavelet
    except ImportError:
        raise ImportError(SCIKIT_IMAGE_REQUIRED)

    if level <= 0:
        raise UserValueError("The level value needs to be a positive integer")
    if level >= len(data.values):
        raise UserValueError("The level value can not exceed the length of data points")
    res = denoise_wavelet(
        data, wavelet_levels=level, wavelet=wavelet, method="VisuShrink", mode="soft", rescale_sigma=True
    )
    return pd.Series(res, index=data.index)
