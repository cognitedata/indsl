import pandas as pd
import pytest

from indsl.sustainability.co2_emissions_calculations import (
    cumulative_co2_cost,
    cumulative_co2_production,
    emissions_factor_combustor,
    rate_of_emissions,
)


@pytest.mark.parametrize(
    ["emissions_factor", "heating_value", "carbon_content", "exp_res"],
    [[25.3, None, None, 25.3], [25.3, 5, None, 126.5], [None, None, 37.7, 37.7]],
)
def test_emissions_factor_combustor_pass(emissions_factor, heating_value, carbon_content, exp_res):
    res = emissions_factor_combustor(
        emissions_factor=emissions_factor, heating_value=heating_value, carbon_content=carbon_content
    )

    assert res == exp_res


@pytest.mark.parametrize(
    ["emissions_factor", "heating_value", "carbon_content"], [[None, None, None], [None, 5, None], [45.3, None, 37.7]]
)
def test_emissions_factor_combustor_fail(emissions_factor, heating_value, carbon_content):
    with pytest.raises(RuntimeError, match=r"Incorrect arguments defined.*"):
        _ = emissions_factor_combustor(
            emissions_factor=emissions_factor, heating_value=heating_value, carbon_content=carbon_content
        )


@pytest.mark.core
def test_rate_of_emissions(test_data):
    res = rate_of_emissions(test_data, 5.67)

    pd.testing.assert_series_equal(res, test_data * 5.67)


@pytest.mark.core
def test_cumulative_co2_production(test_data, test_data_integrated):
    res = cumulative_co2_production(test_data, start_date=pd.Timestamp("1970"))

    pd.testing.assert_series_equal(res, test_data_integrated)


@pytest.mark.core
def test_cumulative_co2_cost(test_data, test_data_integrated):
    res = cumulative_co2_cost(test_data, 5.6, 8.9, start_date=pd.Timestamp("1970"))

    exp_res = test_data_integrated * 5.6 * 8.9

    pd.testing.assert_series_equal(res, exp_res)
