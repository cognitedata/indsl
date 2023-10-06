# Copyright 2021 Cognite AS
import warnings

import pytest

from indsl.forecast.arma_predictor import MethodType, arma_predictor


@pytest.mark.extras
def test_arma_predictor_one_step(create_data_arma):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        # Create data
        perfect_data = create_data_arma[0]
        test_data = create_data_arma[1]

        # Run ARMA smoother
        res = arma_predictor(test_data, steps=1, method=MethodType.ONESTEP, train_fraction=0.8)

        # Check discrepancy, we expect the 2 SSEs to be broadly the same (within 20%)
        res_sse = sum((perfect_data - res).dropna() ** 2)
        data_sse = sum((perfect_data - test_data).tail(21) ** 2)

    assert res_sse < 1.2 * data_sse
    assert test_data.tail(23).shape == res.shape


@pytest.mark.extras
def test_arma_predictor_multi_step(create_data_arma):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        # Create data
        perfect_data = create_data_arma[0]
        test_data = create_data_arma[1]

        # Run ARMA smoother
        res = arma_predictor(test_data, steps=5, method=MethodType.MULTISTEP, train_fraction=0.8)

        # Check discrepancy, we expect the 2 SSEs to be broadly the same (within 20%)
        res_sse = sum((perfect_data - res).dropna() ** 2)
        data_sse = sum((perfect_data.iloc[80:85] - test_data.iloc[80:85]) ** 2)

    assert res_sse < 1.2 * data_sse
    assert test_data.iloc[80:85].shape == res.shape
