# Copyright 2023 Cognite AS
from enum import Enum

import pandas as pd

from indsl import versioning
from indsl.exceptions import SCIKIT_IMAGE_REQUIRED, UserValueError
from indsl.type_check import check_types


class WaveletType(Enum):
    """Wavelet types."""

    DAUBECHIES_1 = "db1"
    DAUBECHIES_2 = "db2"
    DAUBECHIES_3 = "db3"
    DAUBECHIES_4 = "db4"
    DAUBECHIES_5 = "db5"
    DAUBECHIES_6 = "db6"
    DAUBECHIES_7 = "db7"
    DAUBECHIES_8 = "db8"
    SYMLETS_1 = "sym1"
    SYMLETS_2 = "sym2"
    SYMLETS_3 = "sym3"
    SYMLETS_4 = "sym4"
    # TODO: Add more wavelet types from https://pywavelets.readthedocs.io/en/latest/ref/wavelets.html


@versioning.register(version="1.0", deprecated=True)
@check_types
def wavelet_filter(data: pd.Series, level: int = 2, wavelet: WaveletType = WaveletType.DAUBECHIES_8) -> pd.Series:
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
        data, wavelet_levels=level, wavelet=wavelet.value, method="VisuShrink", mode="soft", rescale_sigma=True
    )

    return pd.Series(res, index=data.index)
