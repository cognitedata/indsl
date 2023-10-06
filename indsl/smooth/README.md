# Smoothers

A library of different smoothers, as described below.

## ALMA

ALMA (Arnaud Legoux moving average) is a smoother typically used in the financial industry which aims to
strike a good balance between smoothness and responsivness (i.e., capture a general smoothed trend without
allowing for significant lag). It can be interpreted as a Gaussian weighted moving average with an
offset, where the offset, spread and window size is user-defined.

### Input

- data (pandas.Series): Data to be smoother.
- window (int, optional): Window size. Defaults to 10.
- sigma (float, optional): Parameter that controls the width of the Gaussian filter. Defaults to 6.
- offset_factor (int, optional): Parameter that controls the magnitude of the weights for each past observation within the window. Defaults to 0.75.

### Output

pandas.Series: Smoothed data.

## ARMA SMOOTHER

ARMA (Autoregressive Moving Average) smoother works by fitting constants to a ARMA parametric equation and performing in-sample predictions.

### Input

- data (pandas.Series): Data passed into function.
- ar_order (int, optional): Number of AR terms in the equation. Defaults to 2.
- ma_order (int, optional): Number of MA terms in the equation. Defaults to 2.

### Output

pandas.Series: Smoothed data.

## Butterworth

Butterworth filter.
Uses second-order section (sos) representation, polynomial (ba) representation,
or zeros, poles, and system gain (zpk) representation.

### Input

- data (pandas.DataFrame or pandas.Series): Time series to filter.
- N (int, optional): Order of the filter. Defaults to 50.
- Wn (float, optional): Critical frequency. Defaults to 0.1.
- output (str, optional): Filtering method to apply. Defaults to "sos".
- btype (str, optional): Type of filter - {'lowpass', 'highpass', 'bandpass', 'bandstop'}. Defaults to "lowpass".

### Output

pandas.DataFrame or pandas.Series: Filtered signal.

## Chebyshev

Chebyshev filter of type 1 and 2.
Uses second-order section (sos) output filter type.

### Input

- data (pandas.DataFrame or pandas.Series): Time series to filter.
- filter_type (int, optional): Chebyshev filter type (either 1 or 2). Defaults to 1.
- N (int, optional): Order of the filter. Defaults to 10.
- rp (float, optional): Maximum ripple allowed below unity gain in the passband. Defaults to 0.1.
- Wn (float, optional): Critical frequency. Defaults to 0.1.
- btype (str, optional): Type of filter - {'lowpass', 'highpass', 'bandpass', 'bandstop'}. Defaults to "lowpass".

### Output

pd.DataFrame or pd.Series: Filtered signal.

## Eweight_ma

Exponential Weighted Moving Average.
The exponential moving average gives more weight to the more recent observations. The weights fall exponentially as the data point gets older.
It reacts more than the simple moving average with regards to recent movements.
The moving average value is calculated following the definition yt=(1−α)yt−1+αxt if adjust = False or yt=(xt+(1−α)*xt−1+(1−α)^2*xt−2+...+(1−α)^t\*x0) / (1+(1−α)+(1−α)^2+...+(1−α)^t) if adjust = True.

### Input

- data (pandas.Series): Data with a pd.DatetimeIndex.
- time_window (str, optional): Defines how important the current observation is in the calculation of the EWMA. The longer the period, the slower it adjusts to reflect current trends. Defaults to '60min'.
  If the user gives a number without unit (such as '60'), it will be considered as the number of minutes.
  Accepted string format: '3w', '10d', '5h', '30min', '10s'.
  The time window is converted to the number of points for each of the windows. Each time window may have different number of points if the time series is not regular.
  The number of points specify the decay of the exponential weights in terms of span α=2/(span+1), for span≥1.
- min_periods (int, optional): Minimum number of observations in window required to have a value (otherwise result is NA). Defaults to 1.
  If min_periods > 1 and adjust is False, the SMA is computed for the first observation.
- adjust (bool, optional): If true, the exponential function is calculated using weights wi=(1−α)^i.
  If false, the exponential function is calculated recursively using yt=(1−α)yt−1+αxt. Defaults to True.
- max_pt (int, optional): Sets the maximum number of points to consider in a window if adjust = True. A high number of points will require more time to execute. Defaults to 200.
- resample (bool, optional): Resamples the calculated exponential moving average series. Defaults to False.
- resample_window (str, optional): Time window used to resample the calculated exponential moving average series. Defaults to '60min'.

### Output

pandas.Series: Smoothed time series using exponential weighted moving average.

## lweight_ma

Linear Weighted Moving Average (LWMA)
The linear weighted moving average gives more weight to the more recent observations and gradually less to the older ones.

### Input

data (pandas.Series): Data with a pandas.DatetimeIndex.

- time_window (str, optional): Length of the time period to compute the rolling mean. Defaults to '60min'.
  If the user gives a number without unit (such as '60'), it will be considered as the number of minutes.
  Accepted string format: '3w', '10d', '5h', '30min', '10s'.
- min_periods (int, optional): Minimum number of observations in window required to have a value (otherwise result is NA). Defaults to 1.
- resample (bool, optional): Resamples the calculated linear weighted moving average series. Defaults to False
- resample_window (str, optional): Time window used to resample the calculated linear weighted moving average series. Defaults to '60min'.

### Output

pandas.Series: Smoothed time series using linear weighted moving average.

## Savitzky_golay

Saviztky-Golay data smoothing filter
Use this filter for smoothing data, without distorting the data tendency. This smoothing method is independent of
the sampling frequency. Hence, it is simple and robust to apply to data with non-uniform sampling frequency.
If you are working with high-frequency data (e.g., sampling frequency > 1 Hz), we recommend
selecting the window_length and polyorder parameters manually to suit the requirements. Otherwise, if no parameters
are set, the script will automatically set the parameters based on some of the characteristics of the time series,
such as sampling frequency.

### Input

- series (pandas.Series): The data to be filtered. The series must have a pandas.DatetimeIndex.
- window_length (int, optional): Length of the filter window (i.e. number of data points). A large window results in a stronger smoothing effect and vice-versa.
  If the filter window_length is not defined by the user, a length of about 1/5 of the length of time series is set.
- polyorder (int, optional): The order of the polynomial used to fit the samples. Must be less than window_length.
  Hint: A small polyorder (e.g. polyorder = 1) results in a stringer data smoothing effect. Defaults to 1 if not specified. This typically results in a
  smoothed time series representing the dominating data trend and attenuates the data fluctuations.

### Output

pandas.Series: Smoothed time series with the original timestamp.

## simple_ma

Simple Moving Average (SMA)
Plain simple average that computes the sum of the values of the observations in a time_window divided by the number of observations in the time_window.
SMA time series are much less noisy than the original time series. However, SMA time series lag the original time series, which means that changes in the trend are only seen with a delay (lag) of time_window/2.

### Input

- data (pandas.Series): Data with a pd.DatetimeIndex.
- time_window (str, optional): Length of the time period to compute the average. Defaults to '60min'.
  Accepted string format: '3w', '10d', '5h', '30min', '10s'.
  If the user gives a number without unit (such as '60'), it will be considered as the number of minutes.
- min_periods (int, optional): Minimum number of observations in window required to have a value (otherwise result is NA). Defaults to 1.

### Output

pandas.Series: Smoothed time series using simple moving average.
