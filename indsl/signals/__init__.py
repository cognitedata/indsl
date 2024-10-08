# Copyright 2023 Cognite AS
from .generator import const_value, insert_data_gaps, line, perturb_timestamp, sine_wave, wave_with_brownian_noise
from .noise import white_noise
from .polynomial import univariate_polynomial
from .sequence_interpolation import sequence_interpolation_1d, sequence_interpolation_2d


TOOLBOX_NAME = "Signal generator"

__all__ = [
    "line",
    "perturb_timestamp",
    "insert_data_gaps",
    "sine_wave",
    "white_noise",
    "univariate_polynomial",
    "const_value",
    "wave_with_brownian_noise",
    "sequence_interpolation_1d",
    "sequence_interpolation_2d",
]

__cognite__ = [
    "line",
    "perturb_timestamp",
    "insert_data_gaps",
    "sine_wave",
    "white_noise",
    "univariate_polynomial",
    "const_value",
    "wave_with_brownian_noise",
    "sequence_interpolation_1d",
    "sequence_interpolation_2d",
]
