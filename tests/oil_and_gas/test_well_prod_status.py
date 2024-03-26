import pandas as pd
import pytest

from pandas.testing import assert_series_equal

from indsl.exceptions import UserValueError
from indsl.oil_and_gas.well_prod_status import calculate_well_prod_status, merge_valves, calculate_xmt_prod_status
from datetime import datetime


class TestWellProdStatus:
    @pytest.mark.core
    def test_all_valve_combos(cls):
        # first datapoint is when all valves are open, second is when all
        # valves are closed and last is when only choke is open

        # Inputs
        master = pd.Series([100, 0, 0])
        wing = pd.Series([100, 0, 0])
        choke = pd.Series([100, 0, 100])
        threshold_master = 1
        threshold_wing = 1
        threshold_choke = 5

        res = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)
        exp_res = pd.Series([1, 0, 0]).astype(int)

        assert_series_equal(res.astype(int), exp_res)

    @pytest.mark.core
    def test_high_choke_threshold(cls):
        # test the condition when the choke threshold is very high
        # first test is when choke opening is below threshold

        # Inputs
        master = pd.Series([100])
        wing = pd.Series([100])
        choke = pd.Series([20])
        threshold_master = 1
        threshold_wing = 1
        threshold_choke = 50

        res = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)
        exp_res = pd.Series([0], dtype=int)

        assert_series_equal(res, exp_res)

        # check choke opening above crazy threshold
        choke = pd.Series([100])

        res = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)
        exp_res = pd.Series([1], dtype=int)

        assert_series_equal(res, exp_res)

    @pytest.mark.core
    def test_zero2one_valve_range(cls):
        # test the condition when the choke threshold is very high
        # first test is when choke opening is below threshold

        # Inputs
        master = pd.Series([100, 100, 100, 100, 100])
        wing = pd.Series([0.1, 0.2, 0.3, 0.005, 0.1])
        choke = pd.Series([20, 20, 20, 20, 20])
        threshold_master = 1
        threshold_wing = 1
        threshold_choke = 5

        res = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)
        exp_res = pd.Series([1, 1, 1, 0, 1], dtype=int)

        assert_series_equal(res.astype(int), exp_res)

    @pytest.mark.core
    def test_raise_error_threshold(cls):
        # threshold values should always be greater than or equal to 0

        # Inputs
        master = pd.Series([100])
        wing = pd.Series([100])
        choke = pd.Series([20])
        threshold_master = -1
        threshold_wing = 1
        threshold_choke = 50

        # check if condition statement triggers ValueError
        with pytest.raises(ValueError, match="Threshold value has to be greater than or equal to 0"):
            _ = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)

        # threshold values should always be greater than or equal to 100

        # Inputs
        master = pd.Series([100])
        wing = pd.Series([100])
        choke = pd.Series([20])
        threshold_master = 101
        threshold_wing = 1
        threshold_choke = 50

        # check if condition statement triggers ValueError
        with pytest.raises(ValueError, match="Threshold value has to be less than or equal to 100"):
            _ = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)

    @pytest.mark.core
    def test_empty_valve(cls):
        # check if empty valve raises error

        # Inputs
        master = pd.Series([])
        wing = pd.Series([100])
        choke = pd.Series([20])
        threshold_master = 1
        threshold_wing = 1
        threshold_choke = 50

        # check if condition statement triggers ValueError
        with pytest.raises(UserValueError, match="Empty Series are not allowed for valve inputs"):
            _ = calculate_well_prod_status(master, wing, choke, threshold_master, threshold_wing, threshold_choke)


class TestXmtProdStatus:
    @pytest.mark.core
    def test_merge_valves(cls):
        # test the merging of valves

        t1 = datetime(2019, 1, 1)
        t2 = datetime(2019, 1, 2)
        t3 = datetime(2019, 1, 3)
        t4 = datetime(2019, 1, 4)
        # Inputs
        valve1 = pd.Series([12, 11], index=[t1, t2])
        valve2 = pd.Series([10, 20], index=[t1, t3])
        valve3 = pd.Series([15, 30], index=[t1, t4])

        res = merge_valves(valves=[valve1, valve2, valve3])
        exp_res = pd.Series([10, 11, 20, 30], index=[t1, t2, t3, t4])

        assert_series_equal(res, exp_res, check_dtype=False)

    @pytest.mark.core
    def test_merge_valves_empty_list(cls):
        res = merge_valves(valves=[])
        exp_res = pd.Series([])

        assert_series_equal(res, exp_res, check_dtype=False)

    @pytest.mark.core
    def test_all_valve_combinations(cls):
        t1 = datetime(2019, 1, 1)  # All valves open -> 1
        t2 = datetime(2019, 1, 2)  # Alle wellhead valves open, choke valve closed -> 0
        t3 = datetime(
            2019, 1, 3
        )  # One of two master valves closed, all other wellhead valves open, choke valve open -> 1
        t4 = datetime(
            2019, 1, 4
        )  # One of two master valves closed, all other wellhead valves open, choke valve closed -> 0
        t5 = datetime(2019, 1, 5)  # All wellhead valves closed, choke valve open -> 0

        master_valve1 = pd.Series([100, 100, 100, 100, 0], index=[t1, t2, t3, t4, t5])
        master_valve2 = pd.Series([100, 100, 0, 0, 0], index=[t1, t2, t3, t4, t5])
        annulus_valve = pd.Series([100, 100, 100, 100, 0], index=[t1, t2, t3, t4, t5])
        xover_valve = pd.Series([100, 100, 100, 100, 0], index=[t1, t2, t3, t4, t5])
        choke_valve = pd.Series([100, 0, 20, 0, 50], index=[t1, t2, t3, t4, t5])

        xmt_status = calculate_xmt_prod_status(
            master_valves=[master_valve1, master_valve2],
            annulus_valves=[annulus_valve],
            xover_valves=[xover_valve],
            choke_valve=choke_valve,
        )

        exp_xmt_status = pd.Series([1, 0, 1, 0, 0], index=[t1, t2, t3, t4, t5])
        assert_series_equal(xmt_status, exp_xmt_status, check_dtype=False)

    @pytest.mark.core
    def test_missing_choke_valve(cls):
        t1 = datetime(2019, 1, 1)

        master_valve = pd.Series([100], index=[t1])
        annulus_valve = pd.Series([100], index=[t1])
        xover_valve = pd.Series([100], index=[t1])

        with pytest.raises(TypeError, match="missing 1 required positional argument"):
            _ = calculate_xmt_prod_status(
                master_valves=[master_valve],
                annulus_valves=[annulus_valve],
                xover_valves=[xover_valve],
            )

    @pytest.mark.core
    def test_missing_one_wellhead_valve(cls):
        t1 = datetime(2019, 1, 1)

        master_valve = pd.Series([100], index=[t1])
        choke_valve = pd.Series([100], index=[t1])

        xmt_status = calculate_xmt_prod_status(master_valves=[master_valve], choke_valve=choke_valve)

        exp_xmt_status = pd.Series([1], index=[t1])
        assert_series_equal(xmt_status, exp_xmt_status, check_dtype=False)

    @pytest.mark.core
    def test_missing_all_wellhead_valves(cls):
        t1 = datetime(2019, 1, 1)

        choke_valve = pd.Series([100], index=[t1])

        with pytest.raises(
            UserValueError,
            match=r"At least one of the wellhead valve time series \(master, annulus or xover\) must be provided",
        ):
            _ = calculate_xmt_prod_status(choke_valve=choke_valve)

    @pytest.mark.core
    def test_empty_series(cls):
        t1 = datetime(2019, 1, 1)

        master_valve = pd.Series([])
        choke_valve = pd.Series([100], index=[t1])

        with pytest.raises(UserValueError, match="Empty Series are not allowed for valve inputs"):
            _ = calculate_xmt_prod_status(master_valves=[master_valve], choke_valve=choke_valve)

    @pytest.mark.core
    def test_threshold_values(cls):
        t1 = datetime(2019, 1, 1)

        master_valve = pd.Series([100], index=[t1])
        choke_valve = pd.Series([100], index=[t1])

        with pytest.raises(ValueError, match="Threshold value has to be greater than or equal to 0"):
            _ = calculate_xmt_prod_status(
                master_valves=[master_valve], choke_valve=choke_valve, threshold_master=-1, threshold_choke=50
            )

        with pytest.raises(ValueError, match="Threshold value has to be less than or equal to 100"):
            _ = calculate_xmt_prod_status(
                master_valves=[master_valve], choke_valve=choke_valve, threshold_master=101, threshold_choke=50
            )
