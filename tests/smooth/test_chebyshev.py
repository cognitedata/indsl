import numpy as np
import pandas as pd
import pytest

from indsl.exceptions import UserValueError
from indsl.smooth.chebyshev import chebyshev


@pytest.mark.core
def test_chebyshev_validation():
    data = pd.Series(np.random.randn(10))
    with pytest.raises(UserValueError) as excinfo:
        chebyshev(data, filter_type=3)
    assert "Filter type must be either 1 or 2." in str(excinfo.value)
