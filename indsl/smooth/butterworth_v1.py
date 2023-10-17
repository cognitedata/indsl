# Copyright 2023 Cognite AS
import pandas as pd

from scipy.signal import butter, lfilter, sosfilt, zpk2tf

from indsl import versioning
from indsl.exceptions import UserValueError
from indsl.type_check import check_types


# noinspection SpellCheckingInspection
@versioning.register(version="1.0", deprecated=True)
@check_types
def butterworth(
    data: pd.Series, N: int = 50, Wn: float = 0.1, output: str = "sos", btype: str = "lowpass"
) -> pd.Series:
    """Butterworth.

    This signal processing filter is designed to have a
    frequency response as flat as possible in the passband and roll-offs
    towards zero in the stopband.

    In other words, this filter is designed not to modify much the signal at the
    in the passband and attenuate as much as possible the signal at the stopband. At the moment, only low and high pass
    filtering are supported.

    Args:
        data: Time series.
        N: Order.
            Defaults to 50.
        Wn: Critical frequency.
            Number between 0 and 1, with 1 representing one-half of the sampling rate (Nyquist frequency).
            Defaults to 0.1.
        output: Filtering method
            Defaults to "sos".
        btype: Filter type
            The options are: "lowpass" and "highpass"
            Defaults to "lowpass".

    Returns:
        pandas.Series: Filtered signal.
    """
    if output == "sos":
        # Create Butterworth fi
        filter_output = butter(N=N, Wn=Wn, output=output, btype=btype)
        # Apply second order segments
        filtered = sosfilt(filter_output, data, axis=0)
    elif output == "ba":
        # Create Butterworth filter
        b, a = butter(N=N, Wn=Wn, output=output, btype=btype)
        # Apply filter
        filtered = lfilter(b, a, data, axis=0)
    elif output == "zpk":
        # Create Butterworth filter
        z, p, k = butter(N=N, Wn=Wn, output=output, btype=btype)
        # Return polynomial transfer function representation from zeros and poles
        b, a = zpk2tf(z, p, k)
        # Apply filter
        filtered = lfilter(b, a, data, axis=0)
    else:
        raise UserValueError("output argument is not 'sos', 'ba' or 'zpk'.")

    return pd.Series(filtered, index=data.index)
