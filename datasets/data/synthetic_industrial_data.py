from typing import Tuple

import numpy as np
import pandas as pd

from indsl.signals import insert_data_gaps, line, perturb_timestamp, sine_wave, white_noise


def non_linear_non_stationary_signal(
    seed: int = 89756,
    snr: float = 35,
    fraction: float = 0.35,
    start_timestamp: pd.Timestamp = pd.Timestamp.now() - pd.Timedelta("4.5 days"),
    duration: pd.Timedelta = pd.Timedelta("4.5 days"),
    sample_freq: pd.Timedelta = pd.Timedelta("1m"),
    wave_period: Tuple[str, str, str] = ("6h", "1h", "3h"),
    wave_mean: Tuple[float, float, float] = (0.0, 0.0, 5.0),
    wave_amplitude: Tuple[float, float, float] = (6.8, 100.0, 35.0),
    wave_phase: Tuple[float, float, float] = (0.0, 0.0, np.pi),
    slope: Tuple[float, float, float] = (0.00005, -0.000028, 0.00005),
    intercept: Tuple[float, float, float] = (1.0, 5.0, 0.0),
) -> pd.Series:
    """Non-linear, non-stationary signal.

    Returns a time series composed of 3 oscillatory signals,2 nonlinear trends,
    sensor linear drift (small decrease) and white noise. The signal has non-uniform time stamps and 35% of the data
    is randomly removed to generate data gaps. The data gaps are inserted with a constant seed to have reproducible
    behavior.

    Args:
        seed: Seed number for random number generation
        snr: Signal-to-noise-ratio
        fraction: Fraction of data to remove
        start_timestamp: Start time stamp for the signal
        duration: Total duration of the signal
        sample_freq: Sampling frequency as a time delta
        wave_period: Wave period for each of the oscillatory signals
        wave_mean: Mean for each of the oscillatory signals
        wave_amplitude: Wave amplitude for each of the oscillatory signals
        wave_phase: Wave phase for each of the oscillatory signals
        slope: Slope for each of the trend signals
        intercept: Line intercept for each of the trend signals

    Returns:
        pandas.Series: Time series.
            Non-linear, non-stationary synthetic time series of "industrial" quality.

    """
    end_date = start_timestamp + duration

    # Wave 1: Small amplitude, long wave period
    wave01 = sine_wave(
        start_date=start_timestamp,
        end_date=end_date,
        sample_freq=sample_freq,
        wave_period=pd.Timedelta(wave_period[0]),
        wave_mean=wave_mean[0],
        wave_amplitude=wave_amplitude[0],
        wave_phase=wave_phase[0],
    )
    wave01 = np.exp(wave01)

    # Wave 2: Large amplitude, short wave period
    wave02 = sine_wave(
        start_date=start_timestamp,
        end_date=end_date,
        sample_freq=sample_freq,
        wave_period=pd.Timedelta(wave_period[1]),
        wave_mean=wave_mean[1],
        wave_amplitude=wave_amplitude[1],
        wave_phase=wave_phase[1],
    )

    # Wave 3: Large amplitude, short wave period
    wave03 = sine_wave(
        start_date=start_timestamp,
        end_date=end_date,
        sample_freq=sample_freq,
        wave_period=pd.Timedelta(wave_period[2]),
        wave_mean=wave_mean[2],
        wave_amplitude=wave_amplitude[2],
        wave_phase=wave_phase[2],
    )

    # Trends
    trend_01 = (
        line(
            start_date=start_timestamp,
            end_date=end_date,
            sample_freq=pd.Timedelta("1m"),
            slope=slope[0],
            intercept=intercept[0],
        )
        ** 3
    )

    trend_02 = (
        line(
            start_date=start_timestamp,
            end_date=end_date,
            sample_freq=pd.Timedelta("1m"),
            slope=slope[1],
            intercept=intercept[1],
        )
        ** 5
    )

    drift = line(
        start_date=start_timestamp,
        end_date=end_date,
        sample_freq=pd.Timedelta("1m"),
        slope=slope[2],
        intercept=intercept[2],
    )

    signal = wave01 + wave02 + wave03 + trend_01 + trend_02 - drift
    signal_w_noise = perturb_timestamp(white_noise(signal, snr_db=snr, seed=seed))
    data = insert_data_gaps(signal_w_noise, method="Random", fraction=fraction)

    return data
