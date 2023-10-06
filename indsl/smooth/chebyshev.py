# Copyright 2023 Cognite AS
import pandas as pd

from scipy.signal import cheby1, cheby2, sosfilt

from indsl.exceptions import UserValueError
from indsl.type_check import check_types


# noinspection SpellCheckingInspection


@check_types
def chebyshev(
    data: pd.Series,
    filter_type: int = 1,
    N: int = 10,
    rp: float = 0.1,
    Wn: float = 0.1,
    btype: str = "lowpass",
) -> pd.Series:
    """Chebyshev (I, II).

    Chebyshev filters are analog or digital filters having a steeper roll-off than Butterworth filters, and have
    passband ripple (type I) or stopband ripple (type II). Chebyshev filters have the property that they minimize the
    error between the idealized and the actual filter characteristic over the range of the filter but with ripples in
    the passband (Wikipedia).

    Args:
        data: Time series.
        filter_type: Filter type
            Options are 1 or 2. Defaults to 1.
        N: Order
            Defaults to 10.
        rp: Maximum ripple.
            Maximum ripple allowed below unity gain in the passband.
            Defaults to 0.1.
        Wn: Critical frequency.
            Number between 0 and 1, with 1 representing one-half of the sampling rate (Nyquist frequency).
            Defaults to 0.1.
        btype: Filter type.
            The options are: "lowpass" and "highpass"
            Defaults to "lowpass".

    Returns:
        pandas.Series: Filtered signal
    """
    # TODO: Move function to filter toolbox
    # TODO: Implement band and stop pass types
    # Only type 1 and 2 chebyshev filter exist
    if filter_type not in {1, 2}:
        raise UserValueError("Filter type must be either 1 or 2.")

    if len(data) < 1:
        return data

    # Get type 1 and 2 chebyshev filter
    cheby_filters = {1: cheby1, 2: cheby2}

    # Get filter output
    filter_output = cheby_filters[filter_type](output="sos", N=N, rp=rp, Wn=Wn, btype=btype)

    # Filter the data
    filtered = sosfilt(filter_output, data, axis=0)

    # Return series if that was the input type
    return pd.Series(filtered, index=data.index)
