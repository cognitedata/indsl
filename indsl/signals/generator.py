# Copyright 2023 Cognite AS
import warnings

from typing import Any, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.signals.noise import RedNoise, Sinusoidal, TimeSampler, TimeSeries
from indsl.ts_utils.utility_functions import TimeUnits
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index
from indsl.warnings import IndslUserWarning


@check_types
def line(
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    sample_freq: pd.Timedelta = pd.Timedelta("1 m"),
    slope: float = 0,
    intercept: float = 0,
) -> pd.Series:
    """Line.

    Generate a synthetic time series using the line equation. If no end and/or start dates are given, the default
    signal duration is set to 1 day. If no dates are provided, the end date is set to the current date and time.

    Args:
        start_date: Start date.
            The start date of the time series entered as a string, for example: "1975-05-09 20:09:10", or
            "1975-05-09".
        end_date: End date.
            The end date of the time series entered as a string, for example: "1975-05-09 20:09:10", or
            "1975-05-09".
        sample_freq: Frequency.
            Sampling frequency as a time delta, value and time unit. Defaults to '1 minute'. Valid time units are:

                * ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
                * ‘days’ or ‘day’
                * ‘hours’, ‘hour’, ‘hr’, or ‘h’
                * ‘minutes’, ‘minute’, ‘min’, or ‘m’
                * ‘seconds’, ‘second’, or ‘sec’
                * ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
                * ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’
                * ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’.
        slope: Slope.
            Line slope. Defaults to 0 (horizontal line).
        intercept: Intercept.
             Y-intercept. Defaults to 0.

    Returns:
        pandas.Series: Time series
            Synthetic time series for a line

    """
    idx = _make_index(start=start_date, end=end_date, freq=sample_freq)
    line_data = _time_array(idx) * slope + intercept

    return pd.Series(data=line_data, index=idx, name="Line")


@check_types
def const_value(value: float = 0, timedelta: pd.Timedelta = pd.Timedelta("1 W")) -> pd.Series:
    """Constant value.

    This function generates a horizontal line. The assumptions when generating the horizontal line
    are that the start date is set as "1970-01-01", the end date is set as "now", and the sampling is "1 week".
    If the number of data points generated exceeds 100000, then the start date is moved forward, such that the
    number of data points generated is not greater than 100000 with "1 week" sampling resolution.

    Args:
        value: Value.
            value. Defaults to 0.
        timedelta: Granularity.
            Sampling frequency as a time delta, value, and time unit. Defaults to one week ('1 W'). Valid time units are:

                * ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
                * ‘days’ or ‘day’
                * ‘hours’, ‘hour’, ‘hr’, or ‘h’
                * ‘minutes’, ‘minute’, ‘min’, or ‘m’
                * ‘seconds’, ‘second’, or ‘sec’
                * ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
                * ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’
                * ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’.

    Returns:
        pandas.Series: Time series.
            Synthetic time series for a line

    """
    slope = 0
    idx = _make_index(start="1970-01-01", end="now", freq=timedelta)
    if len(idx) > 1e8:
        idx = idx[len(idx) - 100000 :]

    line_data = _time_array(idx) * slope + value
    return pd.Series(data=line_data, index=idx, name="Constant value")


@check_types
def sine_wave(
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    sample_freq: pd.Timedelta = pd.Timedelta("1 s"),
    wave_period: pd.Timedelta = pd.Timedelta("1 h"),
    wave_mean: float = 0,
    wave_amplitude: float = 1,
    wave_phase: float = 0,
) -> pd.Series:
    """Sine wave.

    Generate a time series for a `sine wave <https://en.wikipedia.org/wiki/Sine_wave>`_ with a given wave period,
    amplitude, phase, and mean value. If no end and/or start dates are given, the default signal duration is set to
    1 day. If no dates are provided, the end date is set to the current date and time.

    Args:
        start_date: Start date
            Date-time string when the time series starts. The date must be a string, for example:
            "1975-05-09 20:09:10".
        end_date: End date
            Date-time string when the time series starts. The date must be a string, for example:
            "1975-05-09 20:09:10".
        sample_freq: Frequency.
            Sampling frequency as a time delta, value, and time unit. Defaults to '1 minute'. Valid time units are:

                * ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
                * ‘days’ or ‘day’
                * ‘hours’, ‘hour’, ‘hr’, or ‘h’
                * ‘minutes’, ‘minute’, ‘min’, or ‘m’
                * ‘seconds’, ‘second’, or ‘sec’
                * ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
                * ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’
                * ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’.
        wave_period: Wave period.
            The time it takes for two successive crests (one wavelength) to pass a specified point. For example, defining
            a wave period of :math:`10 min` will generate one full wave every 10 minutes. The period can not be 0. If
            no value is provided, it is 1 minute. Valid time units are:

                * ‘W’, ‘D’, ‘T’, ‘S’, ‘L’, ‘U’, or ‘N’
                * ‘days’ or ‘day’
                * ‘hours’, ‘hour’, ‘hr’, or ‘h’
                * ‘minutes’, ‘minute’, ‘min’, or ‘m’
                * ‘seconds’, ‘second’, or ‘sec’
                * ‘milliseconds’, ‘millisecond’, ‘millis’, or ‘milli’
                * ‘microseconds’, ‘microsecond’, ‘micros’, or ‘micro’
                * ‘nanoseconds’, ‘nanosecond’, ‘nanos’, ‘nano’, or ‘ns’.
        wave_mean: Mean.
            The wave's mean value. Defaults to 0.
        wave_amplitude: Peak amplitude.
            Maximum absolute deviation from the mean. Defaults to 1.
        wave_phase: Phase.
            Specifies (in radians) where in its cycle the oscillation is at time = 0. When the phase is non-zero, the
            wave is shifted in time. A negative value represents a delay, and a positive value represents an advance.
            Defualts to 0.

    Returns:
        pandas.Series: Sine wave

    """
    if sample_freq.total_seconds() <= 0:
        raise UserValueError(
            f"The sampling frequency must be a value higher than zero. " f"Value provided was {sample_freq}"
        )
    if wave_period.total_seconds() == 0:
        raise UserValueError("The wave period can not be zero")

    idx = _make_index(start=start_date, end=end_date, freq=sample_freq)

    wave_period_ = wave_period / pd.Timedelta(1, unit="s")

    angular_freq = 2 * np.pi * (1 / wave_period_)
    wave_data = wave_amplitude * np.sin(angular_freq * _time_array(idx) + wave_phase) + wave_mean

    return pd.Series(data=wave_data, index=idx, name="Sine wave")


@check_types
def wave_with_brownian_noise(
    duration: int = 14400,
    resolution: float = 0.5,
    percentage: float = 100,
    amplitude: float = 10,
    mean: float = 200,
    frequency: float = 0.04,
    noise: List[int] = [1, 1],
):
    """Wave with brownian noise.

    Sinusoidal signal with brownian noise. The signal has a
    given duration of 4 hours as a default, a resolution of 0.5,
    an amplitude of 10, a mean of 200 and a frequency of 0.04 Hz.


    Args:
        duration: Duration.
            Duration of the time series in seconds. Defaults to 14400.
        resolution: Resolution.
            Frequency resolution. Defaults to 0.5.
        percentage: Percentage.
            Percentage of the time series to keep. Defaults to 100.
        amplitude: Amplitude.
            Amplitude of the wave. Defaults to 10.
        mean: Mean.
            Mean of the wave. Defaults to 200.
        frequency: Frequency.
            Frequency of the wave. Defaults to 0.04 Hz.
        noise: Noise.
            Noise of the wave. Defaults to [1, 1].

    Returns:
        pd.Series:
            Sine wave with brownian noise.
    """
    # Initializing TimeSampler
    time_sampler = TimeSampler(stop_time=duration)
    # Sampling irregular time samples
    time_vector = time_sampler.sample_irregular_time(resolution=resolution, keep_percentage=percentage)

    # Initializing Sinusoidal signal
    sinusoid = Sinusoidal(amplitude=amplitude, frequency=frequency)

    # Initializing Red (Brownian) noise
    red_noise = RedNoise(std=noise[0], tau=noise[1])

    # Initializing TimeSeries class with the signal and noise objects
    timeseries_corr = TimeSeries(sinusoid, noise_generator=red_noise)

    # Sampling using the irregular time samples
    data_points, signals_corr, errors_corr = timeseries_corr.sample(time_vector)
    data_points = data_points + mean

    time_vector = pd.to_datetime(time_vector, unit="s")
    return pd.Series(data=data_points, index=time_vector)


@check_types
def _get_sample_frequency(sample_freq: float, unit: TimeUnits):
    """Get sample frequency.

    Helper method to convert the frequency magnitude and units into a string
    that is then used by other function to generate DatetimeIndex data.

    Args:
        sample_freq: Frequency magnitude
        unit: Time unit
            Valid values "ns|us|ms|s|m|h|D|W". Default is "m" (minutes)

    Returns:
        str: Frequency string

    """
    if unit == "m":
        use_unit = "min"
    elif unit == "s":
        use_unit = "S"
    elif unit == "W":
        use_unit = "D"
        sample_freq = sample_freq * 7
    else:
        use_unit = unit
    return pd.Timedelta(sample_freq, use_unit)


@check_types
def perturb_timestamp(data: pd.Series, magnitude: float = 1) -> pd.Series:
    """Perturb timestamp.

    Perturb the date-time index (timestamp) of the original time series using a normal (Gaussian) distribution
    with a mean of zero and a given standard deviation (magnitude) in seconds.

    Args:
        data: Time series
        magnitude: Magnitude.
            Time delta perturbation magnitude in seconds. Has to be large than 0. Defaults to 1.

    Returns:
        pandas.Series: Time series
            Original signal with a non-uniform time stamp.

    Raises:
        UserTypeError: Only time series with a DateTimeIndex are supported
        UserTypeError: If "magnitude" is not a float
        UserValueError: If "magnitude" is not larger than 0

    """
    validate_series_has_time_index(data)
    if magnitude <= 0:
        raise UserValueError(f"Expected magnitude to be > 0, got {magnitude}")

    time = _time_array(data.index)

    rng = np.random.default_rng(1975)
    time_perturbations = rng.normal(0, magnitude, len(time))
    ts = data.copy()
    ts.index += pd.to_timedelta(time_perturbations, "sec")
    ts.index = ts.index.sort_values()

    return ts.sort_index()


@check_types
def insert_data_gaps(
    data: pd.Series,
    fraction: float = 0.25,
    num_gaps: Optional[int] = None,
    data_buffer: int = 5,
    method: Literal["Random", "Single", "Multiple"] = "Random",
) -> pd.Series:
    """Insert data gaps.

    Method to synthetically remove data, i.e., generate data gaps in a time series. The amount of data points removed
    is defined by the given 'fraction' relative to the original time series.

    Args:
        data: Time series
        fraction: Remove fraction.
            Fraction of data points to remove relative to the original time series. Must be a number higher than 0 and
            lower than 1 (0 < keep < 1). Defaults to 0.25.
        num_gaps: Number of gaps.
            Number of gaps to generate. Only needs to be provided when using the "Multiple" gaps method.
        data_buffer: Buffer.
            Minimum of data points to keep between data gaps and at the start and end of the time series. If the buffer
            of data points is higher than 1% of the number of data points in the time series, the end and start buffer
            is set to 1% of the total available data points.
        method: Method
            This function offers multiple methods to generate data gaps:

                * Random: Removes data points at random locations so that the output time series size is a given
                  fraction  ('Remove fraction') of the original time series. The first and last data points are never
                  deleted. No buffer is set between gaps, only for the start and end of the time series.
                  If the buffer of data points is higher than 1% of the number of data points in the time
                  series, the end and start buffer is set to 1% of the total available data points.
                * Single: Remove consecutive data points at a single location. Buffer data points at the start
                  and end of the time series is kept to prevent removing the start and end of the time series. The
                  buffer is set to the maximum value between 5 data points or 1% of the data points in the signal.
                * Multiple: Insert multiple non-overlapping data gaps at random dates and of random
                  sizes such that the given fraction of data is removed. If the number of gaps is not defined or is
                  less than 2, the function defaults to 2 gaps. To avoid gap overlapping, a minimum of 5 data points are imposed at the signal's start and end and between gaps.

    Returns:
        pandas.Series: Output
            Original time series with synthetically generated data gap(s).

    Raises:
        UserTypeError: data is not a time series
        UserTypeError: fraction is not a number
        UserTypeError: fraction is not a number

    """
    original_length = len(data)
    rng = np.random.default_rng(1975)
    points_to_remove = int(original_length * fraction) + 1
    # 1% Data buffers locations to avoid inserting large data gaps at the start and end of the time series
    buffer1 = max(data_buffer, int(original_length * 0.01))
    if int(original_length * fraction) > original_length - buffer1 * 2 or original_length < 3:
        raise UserValueError(
            f"Not enough data in the time series. The original time series minus the data buffers has "
            f"{original_length} data points. After removing {fraction * 100}% of the data, the "
            f"time series would have {int(original_length * (1 - fraction)) - buffer1 * 2} data points. Please use"
            f"a longer time series or reduce the fraction of data to remove"
        )
    buffer2 = original_length - buffer1
    if method == "Random":
        gap_loc = rng.choice(np.arange(1, original_length - 1), size=points_to_remove, replace=False, shuffle=False)
        return data.loc[data.index.difference(data.index[gap_loc])]
    elif method == "Single":
        gap_loc = int(rng.integers(low=buffer1, high=buffer2, size=1))
        # Move the start of the gap location away from the end of the time series so that the data gap ends right
        # at the buffer
        if gap_loc + points_to_remove > buffer2:
            gap_loc = buffer2 - points_to_remove

        idx = data.index[gap_loc : gap_loc + points_to_remove]
        return data.drop(labels=idx)
    elif method == "Multiple":
        if num_gaps is None or num_gaps < 2:
            num_gaps = 3
        if points_to_remove >= original_length - buffer1 * (num_gaps + 1):
            raise UserValueError(
                "The amount of data to remove exceeds the amount of data point available for creating gaps. Please "
                "reduce the fraction of data to remove or use a time series with more data points"
            )

        # Generate random gap location, sizes and loc ranges that amount to the fraction of data to be removed
        gap_loc = np.sort(rng.integers(low=buffer1, high=buffer2, size=num_gaps))
        gap_size = _random_list_of_integers_that_gives_exact_sum(list_length=num_gaps, sum_output=points_to_remove)
        gap_ranges = np.array(
            list(zip(gap_loc, gap_loc + gap_size - 1))
        )  # BE CAREFUL WITH EDGE RANGE INCLUDING/EXCLUDING LAST DATA POINT

        gap_ranges_ = _handle_overlapping_gaps(gap_ranges=gap_ranges, data_length=len(data))
        idx_loc: np.ndarray = np.array([])
        for ii in range(len(gap_ranges_)):
            idx_loc = np.append(idx_loc, np.arange(gap_ranges_[ii][0], gap_ranges_[ii][1] + 1, dtype=int)).astype(int)
        idx = data.index[idx_loc]
        return data.drop(labels=idx)


@check_types
def _handle_overlapping_gaps(gap_ranges: np.ndarray, data_length: int, buffer: int = 5) -> np.ndarray:
    """Handle overlapping gaps.

    Internal method to handle overlapping gaps. The method calculates the
    number of available space to rearrange gaps in such a way that they do not
    overlap. Then traverses the ranges according to their location from left to
    right (time 0 to end) and estimates how many steps ranges to the left or
    right must be moved to comply with gap generation conditions:

    * The first and last 5 data points of the time series must be preserved
    * A minimum number of data points, defined by `buffer` must exist between adjacent gaps
    * The number of available data point to move ranges around can not exceed the number of overlaps
    """
    new_ranges = np.copy(gap_ranges)
    if len(gap_ranges) < 3:
        raise UserValueError(f"At least three ranges are required. Value provided was {gap_ranges}")
    if np.shape(gap_ranges)[1] != 2:
        raise UserValueError(
            f"Too many values in the ranges array. The shape of the array should "
            f"be (n,2), but we got {np.shape(gap_ranges)}"
        )

    @check_types
    def _calculate_overlaps() -> np.ndarray:
        """Calculate overlaps.

        Calculates the overlap between adjacent intervals as positive numbers and the space between intervals
        as a positive number, enforcing a minimum distance between intervals given by a buffer of data points.
        The overlap is always calculated looking to the left of an interval. For example, if two intervals
        with index locations int1=[10, 15] and int2=[16, 20] an overlap of 5 is assigned to it. If they were
        int1=[10, 15] and int2=[22, 25], an overlap of -1 (or free space of 1 index) would be assigned.
        """
        lin_ranges = np.reshape(new_ranges, (1, np.size(new_ranges))).squeeze()
        lin_ranges = np.insert(lin_ranges, 0, -1)
        lin_ranges = np.append(lin_ranges, data_length)
        overlaps = np.diff(lin_ranges)
        # Delete range lengths and account for buffers
        overlaps = 1 + buffer - np.delete(overlaps, np.arange(1, len(overlaps), 2))
        overlaps[-1] += -1
        tot_overlaps = abs(np.sum(overlaps[np.where(overlaps > 0)]))
        tot_available = np.sum(overlaps[np.where(overlaps < 0)])
        if np.sum(overlaps) > 0:
            raise UserValueError(
                f"The data points overlaps ({tot_overlaps}) is higher than the available space to "
                f"re-arrange the gaps ({tot_available}). Please use a smaller fraction of data to "
                f"remove or a longer time series"
            )
        return overlaps

    overlaps = _calculate_overlaps()

    # Check edge cases and adjust positioning of edge gaps
    ind1 = 0
    ind2 = -1
    while overlaps[ind1] > 0 or overlaps[ind2] > 0:
        if overlaps[ind1] > 0:
            new_ranges[ind1] += overlaps[ind1]
            ind1 += 1
        if overlaps[ind2] > 0:
            new_ranges[ind2] += -overlaps[ind2]
            ind2 += -1
        overlaps = _calculate_overlaps()

    # Adjust intermediate gaps that overlap by moving toe left and right depending on
    # available moves on each direction
    for idx in np.arange(1, len(overlaps) - 1):
        if overlaps[idx] > 0:
            left = -np.sum(overlaps[:idx])
            right = -np.sum(overlaps[idx + 1 :])

            if left >= right and overlaps[idx] <= left:  # Move gap to the left
                # Move intervals to the left if more and enough space available to the left
                count = 1
                while overlaps[idx] > 0:
                    if overlaps[idx - count] < 0:
                        new_ranges[idx - count : idx] += -min(overlaps[idx], -overlaps[idx - count])
                    count += 1
                    overlaps = _calculate_overlaps()

            elif left < right and overlaps[idx] <= right:  # Move gap to the right
                # Move intervals to the right if more and enough space available to the right
                count = 1
                while overlaps[idx] > 0:
                    if overlaps[idx + count] < 0:
                        new_ranges[idx : idx + count] += min(overlaps[idx], -overlaps[idx + count])
                        if idx + count == len(overlaps):
                            new_ranges[idx + count] += min(overlaps[idx], -overlaps[idx + count])

                    count += 1
                    overlaps = _calculate_overlaps()
            elif np.all(np.array([left, right]) < overlaps[idx]) and overlaps[idx] <= left + right:
                # Move adjacent gaps away from current gap
                count = 1
                while overlaps[idx] > 0:
                    # Move gaps on the right
                    if overlaps[idx + count] < 0 and idx + count <= len(overlaps):
                        new_ranges[idx : idx + count] += min(overlaps[idx], -overlaps[idx + count])
                        if idx + count == len(overlaps):
                            new_ranges[idx + count] += min(overlaps[idx], -overlaps[idx + count])

                    # Move gaps on the left
                    if overlaps[idx - count] < 0 and idx - count >= 0:
                        new_ranges[idx - count : idx] += -min(overlaps[idx], -overlaps[idx - count])

                    count += 1

                    overlaps = _calculate_overlaps()

    return new_ranges


@check_types
def _time_array(index: pd.DatetimeIndex) -> np.array:
    """Get time array.

    Convert a pandas DatetimeIndex to a time array in seconds, where t=0 is the start date of the index.

    Args:
        index: Date-time index of a pandas time series.

    Returns:
        numpy.array: Time in seconds representation of the date time index.

    """
    if index.empty:
        raise UserValueError("The date-time index is empty!")
    return (index - index[0]).view(np.int64) / 1e9


@check_types
def _make_index(
    start: Union[pd.Timedelta, pd.Timestamp, str, None] = None,
    end: Union[pd.Timedelta, pd.Timestamp, str, None] = None,
    freq: Any = pd.Timedelta(1, "m"),
) -> pd.DatetimeIndex:
    """Make datetime index.

    Method to generate a datetime index with a uniform time stamp. The time step default is 1 minute and
    a date range (total duration) defaults to 1 day, unless different parameter are provided (start, end, and sample
    frequency).

    Args:
        start: Start date-time
        end: End date-time
        freq: Frequency
            Sample frequency in time units, for example '3 h' represents 3 hours.

    Returns:
        index (pandas.DatetimeIndex): Datetime index

    Raises:
       UserValueError: If one of the inputs is wrong

    """
    # If no end and/or start dates are given, the default signal duration is set to '1 day' and end date to 'now'
    end_date = start_date = None
    if start is None and end is None:
        end = "now"
        end_date = pd.Timestamp(end)
        start_date = end_date - pd.Timedelta("1 day")
    elif start is not None and end is not None:
        end_date = pd.Timestamp(end)
        start_date = pd.Timestamp(start)
        if end_date < start_date:
            # end_date, start_date = start_date, end_date
            raise UserValueError(f"Expected start date before end date, got start={start_date}, end={end_date}")
    elif start is not None and end is None:
        start_date = pd.Timestamp(start)
        end_date = start_date + pd.Timedelta("1 day")
    elif start is None and end is not None:
        end_date = pd.Timestamp(end)
        start_date = end_date - pd.Timedelta("1 day")

    # Catch strange sample frequency inputs
    if not isinstance(freq, pd.Timedelta):
        freq = pd.Timedelta(1, "m")
        warnings.warn(
            "Can't recognize the sample frequency, setting it to the '1 m' default.", category=IndslUserWarning
        )

    if freq.total_seconds() <= 0:
        raise UserValueError(
            f"The sampling frequency must be a value higher than zero. "
            f"The value provided was {freq.total_seconds()}"
        )

    index = pd.date_range(start=start_date, end=end_date, freq=freq)
    return index


@check_types
def _random_list_of_integers_that_gives_exact_sum(list_length: int, sum_output: int) -> list:
    if list_length > sum_output:
        raise UserValueError(
            f"The list length ({list_length}) is larger that the expected sum on interval lengths {sum_output}. "
            f"Use a list length lower than or equal to {sum_output}. "
            f"Note: a list length similar to the expected sum can result in zeros (i.e. no gap assigned to an interval"
        )
    if list_length < 2:
        raise UserValueError(f"The list length must be higher that 2. Value provided = {list_length}")

    # Create an list of a given size and initialize to zero
    int_list = [0] * list_length
    rng = np.random.default_rng(1975)

    # Force sum of list components to be equal to the desired sum output
    for i in range(sum_output):
        # Add 1 at random locations up to desired sum
        int_list[rng.integers(0, sum_output) % list_length] += 1

    return int_list
