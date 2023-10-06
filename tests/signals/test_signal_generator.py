import numpy as np
import pandas as pd
import pytest

from scipy.signal import argrelmin, find_peaks

from indsl.exceptions import UserTypeError, UserValueError
from indsl.signals.generator import (
    _handle_overlapping_gaps,
    _make_index,
    _random_list_of_integers_that_gives_exact_sum,
    _time_array,
    const_value,
    insert_data_gaps,
    line,
    perturb_timestamp,
    sine_wave,
)


test_data_4index = [
    ("1975/5/9", "1975-5-10", pd.Timedelta("1s"), 1.0, pd.Timedelta("1 day")),
    ("1975-5-10", None, pd.Timedelta("1 min"), 60.0, pd.Timedelta("1 day")),
    (None, "1975-5-10", pd.Timedelta("1 min"), 60.0, pd.Timedelta("1 day")),
]

test_data_warns = [
    (None, None, None, 60, pd.Timedelta("1 day")),
    ("1975-5-9", "1975/5/10", ["Some random stuff"], 60, pd.Timedelta("1 day")),  # Wrong sample_frequency input
    ("1975-5-9", "1975/5/10", 9999999, 60, pd.Timedelta("1 day")),  # Wrong sample_frequency input
]


@pytest.mark.parametrize("bad_rate", [(-99999), (None)])
def test_wrong_sample_rate(bad_rate):
    with pytest.warns(UserWarning):
        _make_index(freq=bad_rate)


@pytest.mark.core
def test_start_date_after_end_error():
    with pytest.raises(ValueError):
        start_date = "1975-5-20"
        end_date = "1975/5/9"
        _make_index(start=start_date, end=end_date)


# Check that a Signal is by default initializes to a a 1 day duration with given sampling frequency
@pytest.mark.parametrize(
    "start_date,end_date,sample_rate,expected_mean,expected_dt",
    test_data_4index,
)
def test_index_generator(start_date, end_date, sample_rate, expected_mean, expected_dt):
    # Test that the generated DatetimeIndex has the correct sampling frequency and duration based on input parameters
    idx = _make_index(start=start_date, end=end_date, freq=sample_rate)
    # Convert to time, where t=0 is idx[0]
    time = (idx.to_numpy().astype("int64") - idx[0].value) / 1e9
    assert np.diff(time).mean() == expected_mean
    assert idx[-1] - idx[0] == expected_dt


# Check that a Signal is by default initializes to a a 1 day duration with 1 second sampling frequency when freq is wrong or undefined
@pytest.mark.parametrize(
    "start_date,end_date,sample_rate,expected_mean,expected_dt",
    test_data_warns,
)
def test_sample_freq_warns(start_date, end_date, sample_rate, expected_mean, expected_dt):
    with pytest.warns(UserWarning):
        # Test that the generated DatetimeIndex has the correct sampling frequency and duration based on input parameters
        idx = _make_index(start=start_date, end=end_date, freq=sample_rate)
        # Convert to time, where t=0 is idx[0]
        time = (idx.to_numpy().astype("int64") - idx[0].value) / 1e9
        assert np.diff(time).mean() == expected_mean
        assert idx[-1] - idx[0] == expected_dt


@pytest.mark.core
def test_input_date():
    with pytest.raises(ValueError):
        _make_index(start="1975-5-10", end="1975/5/9", freq="1s")


@pytest.mark.parametrize(
    "start_date,end_date,sample_rate,expected_mean,expected_dt",
    test_data_4index,
)
def test_time_vector_generator(start_date, end_date, sample_rate, expected_mean, expected_dt):
    # Test that the generated time vector (array) has the correct sampling frequency and duration
    # and starts from zero
    tv = _time_array(_make_index(start=start_date, end=end_date, freq=sample_rate))
    assert np.diff(tv).mean() == expected_mean
    assert tv[-1] == expected_dt.total_seconds()
    assert tv[0] == 0


@pytest.mark.core
def test_horizontal_line():
    y_int = 25
    assert line(intercept=y_int, sample_freq=pd.Timedelta("1 m")).mean() == y_int


@pytest.mark.core
def test_sloping_lines():
    m, a = 5, 10
    start_date = pd.Timestamp("1975/5/9")
    end_date = pd.Timestamp("1975-5-10")
    hline1 = line(start_date=start_date, end_date=end_date, sample_freq=pd.Timedelta("1 s"), slope=m, intercept=a)
    hline2 = line(start_date=start_date, end_date=end_date, sample_freq=pd.Timedelta("1 s"), slope=-m, intercept=-a)
    assert (hline1 + hline2).mean() == 0


@pytest.mark.core
def test_perturbate_timestamp():
    mag = pd.Timedelta("3 min").total_seconds() * 0.75  # Magnitude of the perturbation in seconds
    line1 = line(
        start_date=pd.Timestamp("1975/05/09"),
        end_date=pd.Timestamp("1975/05/20"),
        slope=1,
        intercept=5,
        sample_freq=pd.Timedelta("3 m"),
    )
    freq1 = line1.index.to_series().diff().mean().total_seconds()
    line2 = perturb_timestamp(line1, magnitude=mag)
    freq2 = line1.index.to_series().diff().mean().total_seconds()
    # Sample size should not change
    assert len(line1) == len(line2)
    # Average sampling rate should be similar before and after time perturbation, if we have a large enough
    # sample and perturbation magnitude similar to the original sample frequency
    assert np.round(freq1) == np.round(freq2)


@pytest.mark.parametrize("remove,method", [(0.25, "Random"), (0.33, "Single")])
def test_data_gaps(remove, method):
    start = pd.Timestamp("1975/05/09")
    end = pd.Timestamp("1975/05/20")
    line1 = line(start_date=start, end_date=end, slope=1, intercept=5, sample_freq=pd.Timedelta("3 m"))
    ts = insert_data_gaps(data=line1, fraction=remove, method=method)
    assert len(ts) == int(len(line1) * (1 - remove))


@pytest.mark.core
def test_data_gaps_default():
    start = pd.Timestamp("1975/05/09")
    end = pd.Timestamp("1975/05/20")
    line1 = line(start_date=start, end_date=end, slope=1, intercept=5, sample_freq=pd.Timedelta("3 m"))
    ts = insert_data_gaps(data=line1)
    assert len(ts) == int(len(line1) * 0.75)


@pytest.mark.core
def test_random_list_integers_equal_to_sum():
    expected = 50
    l_size = 10
    list_of_ints = _random_list_of_integers_that_gives_exact_sum(l_size, expected)
    assert np.sum(list_of_ints) == expected


@pytest.mark.core
def test_multiple_data_gaps():
    start = pd.Timestamp("1975/05/09")
    end = pd.Timestamp("1975/05/20")
    line1 = line(start_date=start, end_date=end, slope=1, intercept=5, sample_freq=pd.Timedelta("3 m"))
    ts = insert_data_gaps(data=line1, fraction=0.25, num_gaps=3, method="Multiple")
    assert len(ts) == int(len(line1) * 0.75)


left_move = np.array([(10, 20), (50, 57), (55, 58), (64, 70)])  # Overlapping gaps with available moves to the left
mult_left_moves = np.array(
    [(10, 20), (44, 47), (50, 57), (55, 58), (64, 70)]
)  # Overlapping gaps with available moves to the left
mult_left_moves_close = np.array(
    [(31, 41), (44, 47), (50, 57), (55, 58), (64, 70)]
)  # Overlapping gaps close to each other with available moves to the left
mult_right_moves_close = np.array(
    [(5, 8), (15, 19), (19, 23), (25, 31), (28, 34), (57, 63), (73, 75)]
)  # Overlapping gaps with available moves to the right
mult_both_dir_moves = np.array(
    [(11, 16), (27, 33), (33, 36), (35, 40), (39, 42), (42, 47), (62, 68)]
)  # Overlapping gaps with available moves to the left and right
overlaps_equal_moves = np.array(
    [(5, 15), (22, 25), (28, 35), (33, 36), (41, 44), (60, 74)]
)  # Overlaps == Available moves

test_data_mult_gaps = [
    (left_move, np.array([[10, 20], [42, 49], [55, 58], [64, 70]])),
    (mult_left_moves, np.array([[10, 20], [33, 36], [42, 49], [55, 58], [64, 70]])),
    (mult_left_moves_close, np.array([[17, 27], [33, 36], [42, 49], [55, 58], [64, 70]])),
    (mult_right_moves_close, np.array([[5, 8], [15, 19], [25, 29], [35, 41], [47, 53], [59, 65], [73, 75]])),
    (mult_both_dir_moves, np.array([[5, 10], [16, 22], [28, 31], [37, 42], [48, 51], [57, 62], [68, 74]])),
    (overlaps_equal_moves, np.array([[5, 15], [21, 24], [30, 37], [43, 46], [52, 55], [61, 75]])),
]


@pytest.mark.parametrize("test_array,expected", test_data_mult_gaps)
def test_handle_overlaps(test_array, expected):
    data_len = 80
    res = _handle_overlapping_gaps(gap_ranges=test_array, data_length=data_len, buffer=5)
    assert np.array_equal(res, expected)


def _code_that_hits_untested_lines_in_handle_overlaps():
    a = np.array([(10, 19), (20, 25), (26, 29), (64, 70)])
    _handle_overlapping_gaps(a, 40, 2)
    _handle_overlapping_gaps(a, 200, 30)


@pytest.mark.core
def test_sine_wave():
    # The default sine wave parameters must generate X amount of full waves from a zero leve, hence a mean of zero or
    # the defined mean
    assert np.round(sine_wave().mean()) == 0
    assert np.round(sine_wave(wave_mean=10).mean()) == 10

    # Generate a wave signal with a 1 s sampling frequency, total duration of 1 day, a 6 h (0.25D) wave period,
    # amplitude 2, mean 10, and phase pi. The result must be a signal with 4 full waves with 4 local maxima with a
    # value of 12 and 4 local minima of 8.
    wave = sine_wave(
        start_date=pd.Timestamp("1975-05-09"),
        end_date=pd.Timestamp("1975-05-10"),
        wave_amplitude=2,
        wave_mean=10,
        wave_period=pd.Timedelta("0.25 D"),
        wave_phase=np.pi,
    )
    # Find all the local max (peaks) and min (argrelmin). The wave signal generated above has a duration of 24 hours
    # and a wave period of 6 hours. That means that a total of 4 full waves should be present, with exactly four local
    # minima of 8 (mean - amplitude) and four local maxima of 12 (mean + amplitude)
    peaks, _ = find_peaks(wave, distance=60)
    assert wave.iloc[peaks].mean() == 12.0
    assert wave.iloc[argrelmin(wave.values)].mean() == 8.0
    assert np.round(wave.mean()) == 10
    assert len(peaks), len(argrelmin(wave.values)) == 4


@pytest.mark.core
def test_sine_wave_user_error():
    with pytest.raises(UserValueError):
        sine_wave(sample_freq=pd.Timedelta("0 s"))
    with pytest.raises(UserValueError):
        sine_wave(sample_freq=pd.Timedelta("-5 D"))
    with pytest.raises(UserValueError):
        sine_wave(wave_period=pd.Timedelta("0 min"))


@pytest.mark.core
def test_const_value():
    res = const_value()
    assert res.mean() == 0


@pytest.mark.core
def test_validations():
    line1 = line(
        start_date=pd.Timestamp("1975/05/09"),
        end_date=pd.Timestamp("1975/05/10"),
        slope=1,
        intercept=5,
        sample_freq=pd.Timedelta("3 m"),
    )
    with pytest.raises(UserTypeError):
        perturb_timestamp(line1, magnitude="45")  # type: ignore
    with pytest.raises(UserValueError):
        perturb_timestamp(line1, magnitude=0)


@pytest.mark.core
def test_large_index():
    short_line = const_value(timedelta=pd.Timedelta(minutes=0.1))
    assert len(short_line) == 100000


@pytest.mark.core
def test_insert_data_gaps_validation_tests():
    start = pd.Timestamp("1975/05/09")
    end = pd.Timestamp("1975/05/10")
    line1 = line(start_date=start, end_date=end, slope=1, intercept=5, sample_freq=pd.Timedelta("3 h"))

    data_buffer = 10
    original_length = len(line1)
    fraction = 0.2
    buffer1 = max(data_buffer, int(original_length * 0.01))

    with pytest.raises(UserValueError) as excinfo:
        insert_data_gaps(data=line1, data_buffer=data_buffer, fraction=fraction)
    expected = (
        f"Not enough data in the time series. The original time series minus the data buffers has "
        f"{original_length} data points. After removing {fraction * 100}% of the data, the "
        f"time series would have {int(original_length * (1 - fraction)) - buffer1 * 2} data points. Please use"
        f"a longer time series or reduce the fraction of data to remove"
    )
    assert expected in str(excinfo.value)

    line2 = line(start_date=start, end_date=end, slope=1, intercept=5, sample_freq=pd.Timedelta("1 h"))
    expected = (
        "The amount of data to remove exceeds the amount of data point available for creating gaps. Please "
        "reduce the fraction of data to remove or use a time series with more data points"
    )
    with pytest.raises(UserValueError) as excinfo:
        insert_data_gaps(data=line2, data_buffer=data_buffer, fraction=fraction, method="Multiple")
    assert expected in str(excinfo.value)


@pytest.mark.core
def test_handle_overlapping_gaps_validation():
    short_array = np.array([1, 2])
    with pytest.raises(UserValueError) as excinfo:
        _handle_overlapping_gaps(gap_ranges=short_array, data_length=1)
    expected = f"At least three ranges are required. Value provided was {short_array}"
    assert expected in str(excinfo.value)

    wrong_shape_array = np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6]])
    with pytest.raises(UserValueError) as excinfo:
        _handle_overlapping_gaps(gap_ranges=wrong_shape_array, data_length=1)
    expected = (
        f"Too many values in the ranges array. The shape of the array should "
        f"be (n,2), but we got {np.shape(wrong_shape_array)}"
    )
    assert expected in str(excinfo.value)

    array = np.array([(31, 41), (44, 47), (50, 57), (55, 58), (64, 70)])
    too_large_gap_requirement = 40
    with pytest.raises(UserValueError) as excinfo:
        _handle_overlapping_gaps(gap_ranges=array, data_length=80, buffer=too_large_gap_requirement)
    assert (
        "The data points overlaps (193) is higher than the available space to "
        "re-arrange the gaps (0). Please use a smaller fraction of data to "
        "remove or a longer time series"
    ) in str(excinfo.value)


@pytest.mark.core
def test_random_st_of_integers_that_gives_exact_same_sum_validation():
    list_length = 1
    sum_output = 2

    with pytest.raises(UserValueError) as excinfo:
        _random_list_of_integers_that_gives_exact_sum(list_length=list_length, sum_output=sum_output)
    expected = f"The list length must be higher that 2. Value provided = {list_length}"
    assert expected in str(excinfo.value)

    sum_output = 0

    with pytest.raises(UserValueError) as excinfo:
        _random_list_of_integers_that_gives_exact_sum(list_length=list_length, sum_output=sum_output)
    expected = (
        f"The list length ({list_length}) is larger that the expected sum on interval lengths {sum_output}. "
        f"Use a list length lower than or equal to {sum_output}. "
        f"Note: a list length similar to the expected sum can result in zeros (i.e. no gap assigned to an interval"
    )
    assert expected in str(excinfo.value)


@pytest.mark.core
def test_time_array_validation():
    empty_index_series = pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64)
    with pytest.raises(UserValueError) as excinfo:
        _time_array(empty_index_series.index)
    expected = "The date-time index is empty!"
    assert expected in str(excinfo.value)


def test_make_index():
    # test to check for a UserValueError when the frequency has a timedelta of 0
    freq = pd.Timedelta(0)
    expected = (
        f"The sampling frequency must be a value higher than zero. " f"The value provided was {freq.total_seconds()}"
    )
    with pytest.raises(UserValueError) as excinfo:
        _make_index(freq=freq)
    assert expected in str(excinfo.value)
