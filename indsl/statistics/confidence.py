# Copyright 2023 Cognite AS
from typing import Union

import pandas as pd

from indsl.type_check import check_types


@check_types
def bands(data: pd.Series, period: str = "1h", K: float = 2.0, as_json: bool = True) -> Union[str, pd.DataFrame]:
    """Confidence bands.

    Confidence bands, also known as Bollinger Bands, are a statistical characterization of a time series fluctuations.
    The confidence bands display a graphical envelope (upper and lower bands) given by the deviation (expressed by the envelope's width). The envelope width is estimated as a factor of the standard deviation for a given
    time period.

    Two input parameters are required to describe the historical behavior of the data, a time window, N, and a
    multiplication factor, K. The window influences the "responsiveness" of the bands to magnitude and frequency of data
    variations. The multiplication factor influences the width of the envelope.

    The Bollinger Bands consist of an N-period moving average (MA) and upper and lower bands at K times an N-period
    standard deviation above and below the moving average (MA +/- K*stdev).

    Args:
        data: Time series.
        period: Window.
            Window length in seconds. Used to estimate the moving average and standard deviation. Defaults to 3600.
        K: Factor.
            Factor used to estimate the width of the envelope K*stdev. Defaults to 2.
        as_json: JSON?
            Return a json dictionary (True) or a pandas DataFrame (False). Defaults to True.

    Returns:
        JSON or pandas.DataFrame: Time index, moving average, and upper and lower rolling confidence bands.
    """
    # TODO: Add the option of entering a float as period in minutes
    # TODO: Implement left-padding to avoid a result with no data (NaN) at the start of the time series

    res = pd.DataFrame(columns=["avg", "lower", "upper"], index=data.index)

    res.avg = data.rolling(window=period, closed="right").mean()
    res.upper = res.avg + K * data.rolling(window=period, closed="right").std()
    res.lower = res.avg - K * data.rolling(window=period, closed="right").std()
    res = res.dropna()

    # Convert to json oriented by index
    if as_json:
        res = res.to_json(orient="index")

    return res
