# Resampling
Resampling contains a function for both interpolate and resample.


## Interpolate
Interpolates data for uniform timestamps between start and end with specified frequency. A wrapper for the [scipy function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).

### Input
- data (pd.Series): Time series to impute.
- method (str): Specifies the interpolation method:
Follows the methods described in the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).
    * 'linear': linear interpolation.
    * 'ffill': forward filling.
    * 'stepwise': yields same result as ffill.
    * 'zero', ‘slinear’, ‘quadratic’, ‘cubic’: spline interpolation of zeroth, first, second or third order.
-  kind (str): Specifies the kind of returned datapoints. Also described in the [scipy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).
    * 'pointwise': returns the pointwise value of the interpolated function for each timestamp.
    * 'average': returns the average of the interpolated function within each time period. This is the same as CDF average aggregates.
-  granularity (str): Frequency of output e.g. '1s' or '2h'. t min refers to minutes and M to months.
The specified granularity of the output, ie. the interpolated timeseries. Frequency string are described in the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
- bounded (int, optional): 1 To fill in for requested points outside of the data range. 0 to ignore said points.
    Default behaviour is to raise a ValueError if the data range is not within start and end and no outside_fill method is specified.
-  outside_fill (str, optional): Specifies how to fill values outside input data range ('NaN' or 'extrapolate').

### Output
Uniform, interpolated time series with specified frequency/granularity.


## Resample
Resample data for uniform timestamps between start and end with specified frequency. Handles both up and downsample.
Implemented functions are [scipy signal resample](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html), [scipy signal resample](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html#scipy.signal.resample_poly), [pandas resample interpolate](https://pandas.pydata.org/docs/reference/api/pandas.core.resample.Resampler.interpolate.html) and several other functions in [pandas resampling](https://pandas.pydata.org/docs/reference/resampling.html).

The function handles missing data by adding random data with the same distribution/mean, for gaps larger than interpolate_resolution.
Gaps with sizes between interpolate_resolution and ffill_resolution will be interpolated. Gaps smaller than ffill_resolution with be forward filled.


### Input
- data (pd.Series): Time series to impute.
- method (string): Method to resample.
    - "fourier" for Fourier method,
    - "polyphase" for polyphase filtering
    - "interpolate" for linear interpolation when upsampling
    - "min", "max", "sum", "count", "mean" when downsampling
- granularity_current (str): Temporal resolution of uniform time series, before resampling.
    If not specified, the frequency will be implied, which only works if no data is missing.
    Follows Pandas DateTime convention.
- granularity_next (str): Temporal resolution of uniform time series, after resamping. Follows Pandas DateTime convention.
    Either num or granularity_next has to be set.
- num (int): The number of samples in the resampled signal. If this is set, the timedeltas will be infered. A function of the inverse of granularity_next
    Either num or granularity_next has to be set.
- downsampling_factor(int): The downsampling factor for polyphase filtering.
- interpolate_resolution (string): Gaps smaller than this will be interpolated, larger than this will be filled by noise
- ffill_resolution (string): Gaps smaller than this will be forwardfill

### Output
Returns a pandas dataseries with the specified granularity.
