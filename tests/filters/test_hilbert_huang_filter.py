import numpy as np
import pandas as pd

from scipy.stats import shapiro

from indsl.filter import hilbert_huang_transform
from indsl.resample.auto_align import auto_align
from indsl.signals import insert_data_gaps, line, perturb_timestamp, sine_wave, white_noise
from indsl.ts_utils.operators import sub


def generate_synthetic_signal():
    wave = sine_wave(
        start_date=pd.Timestamp("1975-05-09"),
        end_date=pd.Timestamp("1975-05-10"),
        wave_amplitude=2,
        wave_mean=10,
        wave_period=pd.Timedelta("0.25 D"),
        wave_phase=np.pi / 4,
    )
    line_function = line(
        start_date=pd.Timestamp("1975-05-09"),
        end_date=pd.Timestamp("1975-05-10"),
        intercept=10,
        slope=0.0001,
        sample_freq=pd.Timedelta("1 s"),
    )

    return wave + line_function


def test_trend_of_signal():
    signal = generate_synthetic_signal()
    signal_with_noise = white_noise(signal, snr_db=30, seed=42)

    trend = hilbert_huang_transform(signal_with_noise)

    detrended_signal = signal_with_noise - trend

    expected_detrended_signal = signal_with_noise - signal

    # mean is close to zero
    assert abs(np.mean(detrended_signal)) <= 0.001
    assert np.allclose(trend, signal, atol=0.04, rtol=0.04)
    assert np.allclose(detrended_signal, expected_detrended_signal, atol=0.05, rtol=0.05)

    # normality test for detrended signal
    stat, p = shapiro(detrended_signal)
    assert p > 0.05


def test_trend_of_data_with_gaps():
    signal = generate_synthetic_signal()
    signal_with_gaps = insert_data_gaps(signal, num_gaps=4, method="Multiple")
    signal_with_noise = white_noise(signal_with_gaps, snr_db=30, seed=42)

    trend = hilbert_huang_transform(signal_with_noise)

    detrended_signal = signal_with_noise - trend
    expected_detrended_signal = signal_with_noise - signal_with_gaps

    assert abs(np.mean(detrended_signal)) <= 0.001

    # compare data without gap
    assert np.allclose(
        trend["1975-05-09 03:00:00":"1975-05-09 14:00:00"],
        signal["1975-05-09 03:00:00":"1975-05-09 14:00:00"],
        atol=0.04,
        rtol=0.04,
    )
    assert np.allclose(
        detrended_signal["1975-05-09 03:00:00":"1975-05-09 14:00:00"],
        expected_detrended_signal["1975-05-09 03:00:00":"1975-05-09 14:00:00"],
        atol=0.05,
        rtol=0.05,
    )


def test_trend_data_perturb_timestamp():
    # synthetic signal for testing
    signal = generate_synthetic_signal()
    signal_with_noise = white_noise(signal, snr_db=30, seed=42)
    signal_with_perturbation = perturb_timestamp(signal_with_noise)

    trend = hilbert_huang_transform(signal_with_perturbation)
    detrended_signal = sub(signal_with_perturbation, trend, True)

    expected_detrended_signal = sub(signal_with_perturbation, signal, True)

    assert abs(np.mean(expected_detrended_signal)) <= 0.001

    # normality test for detrended signal
    stat, p = shapiro(detrended_signal)
    assert p > 0.05

    assert np.allclose(trend, signal, atol=0.04, rtol=0.04)

    detrended_signal, expected_detrended_signal = auto_align([detrended_signal, expected_detrended_signal])
    assert np.allclose(detrended_signal, expected_detrended_signal, atol=0.05, rtol=0.05)