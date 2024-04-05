import os

from typing import get_args

import numpy as np
import pandas as pd
import pytest

from indsl.exceptions import UserValueError
from indsl.filter.wavelet_filter import wavelet_filter, wavelet_options
from indsl.filter.wavelet_filter_v1 import wavelet_filter as wavelet_filter_v1


@pytest.mark.parametrize("wavelet", get_args(wavelet_options))
@pytest.mark.extras
def test_wavelet_filter_options_run(wavelet):
    try:
        input = pd.Series(np.random.random(100), index=pd.date_range(0, freq="1s", periods=100))
        wavelet_filter(input, wavelet=wavelet, level=1)
    except Exception as e:
        pytest.fail(
            f"Running wavelet_filter with wavelet={wavelet} crashed, even though it should not. Exception details: {e}"
        )


@pytest.mark.parametrize("wavlet_filter_function", [wavelet_filter, wavelet_filter_v1])
@pytest.mark.extras
def test_wavelet_filter(wavlet_filter_function):
    from skimage.restoration import estimate_sigma

    # Arrange
    base_path = "" if __name__ == "__main__" else os.path.dirname(__file__)
    data = pd.read_csv(os.path.join(base_path, "../../datasets/data/vol_flow_rate_m3h.csv"), index_col=0)
    data = data.squeeze()
    data.index = pd.to_datetime(data.index)
    # Act
    with pytest.raises(UserValueError) as excinfo:
        wavlet_filter_function(data, level=0)
    # Assert
    assert "The level value needs to be a positive integer" == str(excinfo.value)

    # Arrange
    level_exceeding_data_points = 3000
    # Act
    with pytest.raises(UserValueError) as excinfo:
        wavlet_filter_function(data, level=level_exceeding_data_points)
    # Assert
    assert "The level value can not exceed the length of data points" == str(excinfo.value)

    # Arrange
    estimate_noise_in_data = estimate_sigma(data)
    filtered_data = wavlet_filter_function(data)
    estimate_noise_in_result = estimate_sigma(filtered_data)
    # Assert
    assert estimate_noise_in_data > estimate_noise_in_result
