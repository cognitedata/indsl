import logging

import numpy as np
import pandas as pd
import pytest

from scipy.stats import anderson, shapiro

from indsl.filter import hilbert_huang_transform
from indsl.resample.auto_align import auto_align
from indsl.signals import insert_data_gaps, line, perturb_timestamp, sine_wave, white_noise
from indsl.ts_utils.operators import sub
from sparse import COO


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
    signal_with_noise = white_noise(signal, snr_db=10, seed=42)

    trend = hilbert_huang_transform(signal_with_noise)

    detrended_signal = signal_with_noise - trend

    expected_detrended_signal = signal_with_noise - signal

    # mean is close to zero
    assert abs(np.mean(detrended_signal)) <= 0.05
    # assert np.allclose(trend, signal, atol=0.04, rtol=0.04)
    try:
        assert np.allclose(
            trend["1975-05-09 03:00:00":"1975-05-09 14:00:00"],
            signal["1975-05-09 03:00:00":"1975-05-09 14:00:00"],
            atol=0.4,
            rtol=0.4,
        )
    except AssertionError:
        logging.error("trend and signal are not close. Max difference: %s", np.max(np.abs(trend - signal)))
        raise
