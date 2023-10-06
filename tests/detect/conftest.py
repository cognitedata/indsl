# Copyright 2021 Cognite AS
from datetime import datetime

import numpy as np
import pytest


@pytest.fixture
def sin_wave_array():
    return np.sin(np.linspace(0, 10, 100))


@pytest.fixture
def multi_freq_signal():
    low_freq = 0.02
    high_freq = 0.1
    t = np.linspace(0, 1000, 1000)  # 1000 s

    sig_low = np.sin(2 * np.pi * low_freq * t)
    sig_high = np.sin(2 * np.pi * high_freq * t)

    sig = sig_low * 1 + sig_high * 0.5
    return sig


@pytest.fixture
def sin_wave_array_extensive():
    return np.sin(np.linspace(0, 10, 1000))


@pytest.fixture
def start_date():
    return datetime(2020, 7, 23, 15, 27, 0)


@pytest.fixture
def frequency():
    return "1s"
