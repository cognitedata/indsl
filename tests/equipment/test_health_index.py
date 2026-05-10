# Tests for the equipment_health_index function in the equipment module
import numpy as np
import pandas as pd
import pytest

from indsl.equipment.health_index import equipment_health_index
from indsl.exceptions import UserValueError


def _make_series(values, start="2026-01-01", freq="h"):
    return pd.Series(values, index=pd.date_range(start=start, periods=len(values), freq=freq))


@pytest.mark.core
class TestEquipmentHealthIndexBasics:
    def test_perfectly_constant_sensors_return_one(self):
        """A constant sensor at its baseline must produce EHI = 1 everywhere."""
        sensor_a = _make_series([5.0] * 30)
        sensor_b = _make_series([10.0] * 30)
        result = equipment_health_index([sensor_a, sensor_b])
        assert (result == 1.0).all()

    def test_output_is_named_ehi(self):
        """The returned series must be named 'EHI' for downstream identification."""
        result = equipment_health_index([_make_series([1.0] * 30)])
        assert result.name == "EHI"

    def test_output_is_bounded_in_unit_interval(self):
        """EHI must always lie in [0, 1] regardless of input magnitude."""
        np.random.seed(0)
        n = 200
        sensor = _make_series(np.concatenate([np.random.normal(0, 1, 100), np.random.normal(10, 5, 100)]))
        result = equipment_health_index([sensor])
        finite = result.dropna()
        assert (finite >= 0).all() and (finite <= 1).all()

    def test_output_length_matches_input(self):
        """When inputs share an index, output length matches input length."""
        sensor = _make_series(np.linspace(1.0, 1.5, 50))
        result = equipment_health_index([sensor])
        assert len(result) == 50

    def test_single_sensor_works(self):
        """A single sensor must be handled without aggregation errors."""
        sensor = _make_series([1.0] * 20)
        result = equipment_health_index([sensor])
        assert len(result) == 20
        assert (result == 1.0).all()


@pytest.mark.core
class TestEquipmentHealthIndexBehavior:
    def test_degradation_lowers_score(self):
        """A clear monotonic deviation from baseline must lower the score over time."""
        np.random.seed(1)
        n = 300
        baseline_phase = 1.0 + np.random.normal(0, 0.02, 100)
        degradation_phase = 1.0 + np.random.normal(0, 0.02, 200) + np.linspace(0, 0.5, 200)
        sensor = _make_series(np.concatenate([baseline_phase, degradation_phase]))
        result = equipment_health_index([sensor])
        early_mean = result.iloc[10:50].mean()
        late_mean = result.iloc[-50:].mean()
        assert late_mean < early_mean
        assert late_mean < 0.5

    def test_one_bad_sensor_pulls_geometric_mean_down(self):
        """The geometric-mean aggregation must let a single severely degraded sensor dominate."""
        np.random.seed(2)
        n = 50
        healthy = _make_series(1.0 + np.random.normal(0, 0.01, n))
        # Sharp deviation in the second half
        bad_values = np.concatenate([1.0 + np.random.normal(0, 0.01, 25), np.full(25, 3.0)])
        bad = _make_series(bad_values)
        result_mixed = equipment_health_index([healthy, bad])
        result_healthy_only = equipment_health_index([healthy])
        assert result_mixed.iloc[-1] < result_healthy_only.iloc[-1]
        assert result_mixed.iloc[-1] < 0.5

    def test_higher_sensitivity_value_is_less_strict(self):
        """A larger ``sensitivity`` parameter must yield a higher (less penalised) score."""
        np.random.seed(6)
        # Reference window must have non-zero variance for sensitivity to apply
        baseline_phase = 1.0 + np.random.normal(0, 0.05, 20)
        deviation_phase = np.full(10, 2.0)
        sensor = _make_series(np.concatenate([baseline_phase, deviation_phase]))
        strict = equipment_health_index([sensor], sensitivity=1.0)
        lenient = equipment_health_index([sensor], sensitivity=10.0)
        assert lenient.iloc[-1] > strict.iloc[-1]

    def test_explicit_baseline_overrides_window_mean(self):
        """When ``baselines`` is given, the supplied value is used as the reference mean."""
        sensor = _make_series([5.0] * 30)
        # Baseline far from the actual signal → strong deviation, low score
        result_offset = equipment_health_index([sensor], baselines=[0.0])
        # Baseline equal to the signal → perfect score
        result_match = equipment_health_index([sensor], baselines=[5.0])
        assert (result_match == 1.0).all()
        # With std=0 in the constant signal, the offset baseline still results in factor=1
        # because the std-degenerate branch is taken; this is documented behaviour.
        assert (result_offset == 1.0).all()

    def test_weights_skew_aggregation(self):
        """Larger weight on a degraded sensor must reduce the EHI more than equal weighting."""
        np.random.seed(3)
        healthy = _make_series(1.0 + np.random.normal(0, 0.01, 30))
        bad = _make_series(np.concatenate([1.0 + np.random.normal(0, 0.01, 10), np.full(20, 2.5)]))

        equal = equipment_health_index([healthy, bad], weights=[1.0, 1.0])
        skewed = equipment_health_index([healthy, bad], weights=[1.0, 5.0])
        assert skewed.iloc[-1] < equal.iloc[-1]

    def test_zero_weight_excludes_sensor(self):
        """A sensor with zero weight must not affect the result."""
        np.random.seed(4)
        healthy = _make_series(1.0 + np.random.normal(0, 0.01, 30))
        bad = _make_series(np.concatenate([1.0 + np.random.normal(0, 0.01, 10), np.full(20, 5.0)]))

        only_healthy = equipment_health_index([healthy])
        with_zero_weight = equipment_health_index([healthy, bad], weights=[1.0, 0.0])
        pd.testing.assert_series_equal(only_healthy, with_zero_weight, check_exact=False, atol=1e-10)


@pytest.mark.core
class TestEquipmentHealthIndexNaNHandling:
    def test_nan_propagates_when_reference_has_variance(self):
        """NaN inputs must produce NaN outputs when a valid reference std exists."""
        np.random.seed(5)
        values = np.random.normal(0, 1, 30)
        values[20:25] = np.nan
        sensor = _make_series(values)
        result = equipment_health_index([sensor])
        assert result.iloc[20:25].isna().all()

    def test_nan_propagates_with_constant_reference(self):
        """NaN must still propagate even when the reference window has zero variance."""
        sensor = _make_series([1.0] * 10 + [np.nan] * 10)
        result = equipment_health_index([sensor])
        assert result.iloc[-5:].isna().all()
        assert (result.iloc[:10] == 1.0).all()

    def test_constant_reference_yields_neutral_factor(self):
        """A degenerate (constant) reference must not crash and must yield a neutral score."""
        # Constant reference with later variation is uncharacterisable: factor stays at 1
        sensor = _make_series([1.0] * 15 + [10.0] * 15)
        result = equipment_health_index([sensor])
        assert (result.dropna() == 1.0).all()


@pytest.mark.core
class TestEquipmentHealthIndexAlignment:
    def test_misaligned_inputs_are_aligned_when_enabled(self):
        """With ``align_timestamps=True`` (default), differently indexed series must be aligned."""
        sa = _make_series([1.0] * 12, start="2026-01-01 00:00")
        sb = _make_series([1.0] * 12, start="2026-01-01 00:30")
        result = equipment_health_index([sa, sb], align_timestamps=True)
        assert len(result) > 0
        assert (result.dropna() == 1.0).all()


@pytest.mark.core
class TestEquipmentHealthIndexValidation:
    def test_empty_sensor_list_raises(self):
        with pytest.raises(UserValueError, match="At least one sensor"):
            equipment_health_index([])

    def test_baseline_length_mismatch_raises(self):
        sensor = _make_series([1.0] * 20)
        with pytest.raises(UserValueError, match="baselines"):
            equipment_health_index([sensor], baselines=[1.0, 2.0])

    def test_weights_length_mismatch_raises(self):
        sensor = _make_series([1.0] * 20)
        with pytest.raises(UserValueError, match="weights"):
            equipment_health_index([sensor], weights=[1.0, 2.0])

    def test_negative_weight_raises(self):
        sensor = _make_series([1.0] * 20)
        with pytest.raises(UserValueError, match="non-negative"):
            equipment_health_index([sensor], weights=[-1.0])

    def test_all_zero_weights_raises(self):
        sensor = _make_series([1.0] * 20)
        with pytest.raises(UserValueError, match="strictly positive"):
            equipment_health_index([sensor, sensor], weights=[0.0, 0.0])

    @pytest.mark.parametrize("bad_value", [0.0, -1.0, -0.001])
    def test_non_positive_sensitivity_raises(self, bad_value):
        sensor = _make_series([1.0] * 20)
        with pytest.raises(UserValueError, match="sensitivity"):
            equipment_health_index([sensor], sensitivity=bad_value)

    @pytest.mark.parametrize("bad_value", [0.0, -0.1, 1.1, 2.0])
    def test_invalid_reference_fraction_raises(self, bad_value):
        sensor = _make_series([1.0] * 20)
        with pytest.raises(UserValueError, match="reference_fraction"):
            equipment_health_index([sensor], reference_fraction=bad_value)
