# Copyright 2021 Cognite AS
import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.data_quality import extreme
from indsl.exceptions import UserValueError
from tests.detect.test_utils import RNGContext


@pytest.mark.core
def test_anomaly_detector():
    # Create fake data
    with RNGContext():
        sig = np.random.normal(loc=100, size=1000, scale=1)
    index = pd.date_range(start="1970", periods=len(sig), freq="1T")

    # Add anomalies
    anom_num = [61, 61, 126, 83, 133, 21, 126, 126, 187, 13, 86, 140, 44, 146, 89, 135, 52, 31, 43, 197]
    anom_ids = [886, 427, 304, 871, 970, 316, 962, 634, 901, 8, 773, 72, 358, 222, 552, 369, 715, 263, 839, 689]

    sig[anom_ids] = anom_num

    clean_sig = np.delete(sig, anom_ids)

    anomalous_data = pd.Series(sig, index=index)
    clean_data = pd.Series(clean_sig, index=np.delete(index, anom_ids))

    # Call the anomaly detector
    res = extreme(anomalous_data)

    # Check results (all anomalies should be removed)
    assert_series_equal(res, clean_data)

    with pytest.raises(
        UserValueError, match="The significance level must be a number higher than or equal to 0 and lower than 1"
    ):
        _ = extreme(anomalous_data, alpha=-1)
        _ = extreme(anomalous_data, alpha=2)
