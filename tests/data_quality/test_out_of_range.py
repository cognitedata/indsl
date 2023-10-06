# Copyright 2022 Cognite AS

import pytest

from indsl.data_quality.outliers import _validations, out_of_range
from indsl.exceptions import UserValueError


@pytest.mark.core
def test_out_of_range_find_outliers(generate_synthetic_industrial_data_with_outliers):
    """Test for Out of Range data.

    The fixture generates 5 non-linear, non-stationary signals, of similar characteristics, with a fraction (5%)
    being extreme outliers.

    This test confirms that the out_of_range script identifies ALL the outliers in each case.
    """
    data, outliers = generate_synthetic_industrial_data_with_outliers
    res = out_of_range(data)
    assert res.equals(outliers)


@pytest.mark.core
def test_validations():
    window_length = [20, 20]
    polyorder = [3, 3]
    alpha = [0.05, -2]
    bc_relaxation = [0.25, 0.5]
    with pytest.raises(UserValueError) as excinfo:
        _validations(
            alpha=alpha,
            bc_relaxation=bc_relaxation,
            window_length=window_length,
            polyorder=polyorder,
        )
    assert "The Significance Level (alpha) must be a number higher than or equal to 0 and lower " "than 1" == str(
        excinfo.value
    )

    window_length = [20, 20]
    polyorder = [3, 3]
    alpha = [0.05, 0.05]
    bc_relaxation = [0.25, -0.5]
    with pytest.raises(UserValueError) as excinfo:
        _validations(
            alpha=alpha,
            bc_relaxation=bc_relaxation,
            window_length=window_length,
            polyorder=polyorder,
        )
    assert "The Relaxation Factor must be a number higher than 0" == str(excinfo.value)

    window_length = [20, 20, 20]
    polyorder = [3, 3]
    alpha = [0.05, 0.05]
    bc_relaxation = [0.25, 0.5]
    with pytest.raises(UserValueError) as excinfo:
        _validations(
            alpha=alpha,
            bc_relaxation=bc_relaxation,
            window_length=window_length,
            polyorder=polyorder,
        )
    assert "The window length requires two values, got [20, 20, 20]" == str(excinfo.value)

    window_length = [20, 20]
    polyorder = [3, 3, 3]
    alpha = [0.05, 0.05]
    bc_relaxation = [0.25, 0.5]
    with pytest.raises(UserValueError) as excinfo:
        _validations(
            alpha=alpha,
            bc_relaxation=bc_relaxation,
            window_length=window_length,
            polyorder=polyorder,
        )
    assert "The polynomial order parameter requires two values, got [3, 3, 3]" == str(excinfo.value)

    window_length = [20, 20]
    polyorder = [3, 3]
    alpha = [0.05, 0.05, 0.05]
    bc_relaxation = [0.25, 0.5]
    with pytest.raises(UserValueError) as excinfo:
        _validations(
            alpha=alpha,
            bc_relaxation=bc_relaxation,
            window_length=window_length,
            polyorder=polyorder,
        )
    assert "The Significance Level (alpha) parameter requires two values, got [0.05, 0.05, 0.05]" == str(excinfo.value)

    window_length = [20, 20]
    polyorder = [3, 3]
    alpha = [0.05, 0.05]
    bc_relaxation = [0.25, 0.5, 0.5]
    with pytest.raises(UserValueError) as excinfo:
        _validations(
            alpha=alpha,
            bc_relaxation=bc_relaxation,
            window_length=window_length,
            polyorder=polyorder,
        )
    assert "The sensitivity parameter requires two values, got [0.25, 0.5, 0.5]" == str(excinfo.value)
