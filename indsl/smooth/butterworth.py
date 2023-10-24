# Copyright 2023 Cognite AS
from typing import Literal

import pandas as pd

from scipy.signal import butter, sosfilt

from indsl import versioning
from indsl.type_check import check_types

from . import butterworth_v1  # noqa


# noinspection SpellCheckingInspection
@versioning.register(
    version="2.0",
    changelog="Unused or irrelevant lines of code removed and filter output parameter "
    "removed from function signature and set to `sos`.",
)
@check_types
def butterworth(
    data: pd.Series,
    N: int = 50,
    Wn: float = 0.1,
    btype: Literal["lowpass", "highpass"] = "lowpass",
) -> pd.Series:
    """Butterworth.

    This signal processing filter is designed to have a frequency response as flat as possible in the passband and
    roll-offs towards zero in the stopband. In other words, this filter is designed not to modify much the signal at the
    in the passband and attenuate as much as possible the signal at the stopband. At the moment, only low and high pass
    filtering are supported.

    Args:
        data: Time series.
        N: Order.
            Defaults to 50.
        Wn: Critical frequency.
            Number between 0 and 1, with 1 representing one-half of the sampling rate (Nyquist frequency).
            Defaults to 0.1.
        btype: Filter type.
            The options are: "lowpass" and "highpass"
            Defaults to "lowpass".

    Returns:
        pandas.Series: Filtered signal.
    """
    data = data.dropna()

    if len(data) < 1:
        return data

    filter_output = butter(N=N, Wn=Wn, output="sos", btype=btype)
    # Apply second order segments
    filtered = sosfilt(filter_output, data, axis=0)

    return pd.Series(filtered, index=data.index)
