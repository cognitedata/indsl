# Copyright 2026 Cognite AS
from typing import List, Optional

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types


@check_types
def equipment_health_index(
    sensors: List[pd.Series],
    baselines: Optional[List[float]] = None,
    weights: Optional[List[float]] = None,
    sensitivity: float = 3.0,
    reference_fraction: float = 0.1,
    align_timestamps: bool = True,
) -> pd.Series:
    r"""Equipment health index (EHI).

    Computes a unified health score in :math:`[0, 1]` from multiple sensor
    signals, where ``1`` indicates a perfectly healthy state matching the
    reference baseline and values approaching ``0`` indicate severe deviation.

    The score is computed in two steps. For each sensor :math:`i`, an absolute
    z-score is calculated against the baseline mean :math:`\mu_i` and reference
    standard deviation :math:`\sigma_i`,

    .. math:: z_i(t) = \frac{|x_i(t) - \mu_i|}{\sigma_i},

    and converted to a per-sensor health factor bounded in :math:`(0, 1]`,

    .. math:: f_i(t) = \exp\!\left(-\frac{z_i(t)}{s}\right),

    where :math:`s` is the ``sensitivity`` parameter. The final health index is
    the weighted geometric mean of the per-sensor factors,

    .. math:: \mathrm{EHI}(t) = \exp\!\left(\frac{\sum_i w_i \ln f_i(t)}{\sum_i w_i}\right).

    The geometric mean is used so that any single severely degraded sensor
    pulls the overall score down, which matches the conservative
    interpretation typically used in condition monitoring.

    Args:
        sensors: Sensor time series.
            List of one or more sensor signals. Series do not need to share
            an index when ``align_timestamps`` is ``True`` (the default).
        baselines: Reference baseline values.
            One value per sensor representing the healthy reference. If not
            provided, the mean of the first ``reference_fraction`` of each
            series is used.
        weights: Sensor weights.
            One non-negative weight per sensor. Defaults to equal weighting.
        sensitivity: Sensitivity factor.
            The z-score divisor controlling how aggressively deviations
            reduce the health score. Larger values are less sensitive.
            Must be strictly positive. Defaults to ``3.0``.
        reference_fraction: Reference window fraction.
            Fraction of the leading samples of each series used to estimate
            the baseline mean (when not supplied) and the reference standard
            deviation. Must lie in ``(0, 1]``. A minimum of ten samples is
            always used. Defaults to ``0.1``.
        align_timestamps: Auto-align.
            Whether to automatically align the input sensor series to a
            common index. Defaults to ``True``.

    Returns:
        pd.Series: Equipment health index.
            Time series in :math:`[0, 1]` named ``"EHI"`` and indexed by the
            common (aligned) timestamps of the inputs.

    Raises:
        UserValueError: If no sensor is provided, if the lengths of
            ``baselines`` or ``weights`` do not match the number of sensors,
            if any weight is negative or all weights are zero, if
            ``sensitivity`` is non-positive, or if ``reference_fraction``
            is outside ``(0, 1]``.
    """
    # ---- Input validation ----------------------------------------------------
    if len(sensors) == 0:
        raise UserValueError("At least one sensor series must be provided.")
    if baselines is not None and len(baselines) != len(sensors):
        raise UserValueError(f"Length of `baselines` ({len(baselines)}) must match number of sensors ({len(sensors)}).")
    if weights is not None:
        if len(weights) != len(sensors):
            raise UserValueError(f"Length of `weights` ({len(weights)}) must match number of sensors ({len(sensors)}).")
        if any(w < 0 for w in weights):
            raise UserValueError("All weights must be non-negative.")
        if sum(weights) == 0:
            raise UserValueError("At least one weight must be strictly positive.")
    if sensitivity <= 0:
        raise UserValueError(f"`sensitivity` must be strictly positive, got {sensitivity}.")
    if not (0 < reference_fraction <= 1):
        raise UserValueError(f"`reference_fraction` must lie in (0, 1], got {reference_fraction}.")

    # ---- Align inputs --------------------------------------------------------
    aligned: List[pd.Series] = auto_align(sensors, enabled=align_timestamps)

    effective_weights = weights if weights is not None else [1.0] * len(aligned)
    weight_sum = float(sum(effective_weights))

    # ---- Per-sensor health factor and weighted log accumulation --------------
    log_acc: Optional[pd.Series] = None

    for i, series in enumerate(aligned):
        n_ref = max(10, int(reference_fraction * len(series)))
        n_ref = min(n_ref, len(series))
        reference_window = series.iloc[:n_ref]

        baseline_mean = baselines[i] if baselines is not None else float(reference_window.mean())
        reference_std = float(reference_window.std(ddof=0))

        if not np.isfinite(reference_std) or reference_std == 0.0:
            # Reference window has no characterisable variability (constant or
            # entirely missing). The sensor cannot contribute deviation
            # information, so its factor is set to 1 wherever the signal is
            # observed and to NaN wherever the input is NaN, preserving
            # missingness in the aggregated index.
            factor = pd.Series(np.where(series.notna(), 1.0, np.nan), index=series.index)
        else:
            z_score = (series - baseline_mean).abs() / reference_std
            factor = np.exp(-z_score / sensitivity).clip(lower=1e-6, upper=1.0)

        log_term = effective_weights[i] * np.log(factor)
        log_acc = log_term if log_acc is None else log_acc + log_term

    # log_acc is guaranteed non-None because len(sensors) > 0
    assert log_acc is not None
    ehi = np.exp(log_acc / weight_sum)
    ehi.name = "EHI"
    return ehi
