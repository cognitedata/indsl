# Copyright 2023 Cognite AS
import os
import warnings

from typing import List, Literal, Union

import numpy as np
import pandas as pd

from indsl.exceptions import FLUIDS_REQUIRED
from indsl.resample.auto_align import auto_align
from indsl.type_check import check_types
from indsl.warnings import IndslUserWarning


NUMBA_DISABLED = os.environ.get("NUMBA_DISABLE_JIT") == "1"
# workaround since numba does not allow lazy import
if NUMBA_DISABLED:
    try:
        import fluids.vectorized as fv
    except ImportError:
        fv = None  # Only core dependencies available. Will raise an error later

else:
    try:
        # Will fail if ipython is not installed (and we don't want to add it as a dependency)
        import fluids.numba_vectorized as fv
    except ImportError:
        try:
            import fluids.vectorized as fv
        except ImportError as e:
            warnings.warn(
                f"Couldn't import fluids.numba_vectorized: {e!s}. Default to import fluids.vectorized.",
                category=IndslUserWarning,
            )
            fv = None  # Only core dependencies available. Will raise an error later


def __take_array_or_float(input: Union[pd.Series, float]) -> Union[np.ndarray, float]:
    """Checks if the input is a float or a pandas series and returns the numeric value (float or array).

    Args:
        input (Union[pd.Series, float]): input data.

    Returns:
        Union[np.ndarray, float]: numeric value(s).
    """
    return input.values.astype("float64") if isinstance(input, pd.Series) else input


def __create_series_with_index_of_inputs(inputs: List[Union[pd.Series, float]]) -> pd.Series:
    """Tries to extract the index of one of the input values.

    Args:
        inputs (list[Union[pd.Series, float]]): list of input values.

    Returns:
        Optional[pd.Series]: pandas series with the index of one of the inputs or None.
    """
    # we can take the index of anyone of the time series present in the inputs as they have been aligned already
    for input in inputs:
        if isinstance(input, pd.Series):
            return pd.Series(index=input.index, name="Volume", dtype="float64")


@check_types
def filled_volume_ellipsoidal_head_vessel(
    D: Union[pd.Series, float],
    L: Union[pd.Series, float],
    a: Union[pd.Series, float],
    h: Union[pd.Series, float],
    orientation: Literal["Horizontal", "Vertical"] = "Horizontal",
) -> Union[pd.Series, float]:
    r"""Vessel volume (Ellipsoidal).

    Calculates partially full volume of a vertical or horizontal vessel with ellipsoidal convex heads. For vertical
    vessels the bottom is considered as ellipsoidal, but no provision for the top of the vessel is made here.

    For 2:1 Elliptical Head, :math:`a = D/4`.

    Args:
        D: Internal diameter [m].
            Diameter of the main cylindrical section.
        L: Straight length [m].
            Length of the main cylindrical section.
        a: Inside dish depth (a) [m].
            Distance the ellipsoidal head extends on one side
        h: Fluid level [m].
            Height, as measured from the lowest point of the vessel to the fluid surface.
        orientation: Vessel orientation.

    Returns:
        Union[pd.Series, float]: Volume [m³]
            Volume of liquid in the vessel.

    References:
        Caleb Bell (2016-2021). fluids: Fluid dynamics component of Chemical Engineering Design Library (ChEDL)
        https://github.com/CalebBell/fluids.
    """
    if fv is None:
        raise ImportError(FLUIDS_REQUIRED)

    D, L, a, h = auto_align([D, L, a, h])

    if orientation == "Horizontal":
        # V_horiz_ellipsoidal(D, L, a, h, headonly=False)
        V = fv.V_horiz_ellipsoidal(
            __take_array_or_float(D),
            __take_array_or_float(L),
            __take_array_or_float(a),
            __take_array_or_float(h),
            False,
        )
    elif orientation == "Vertical":
        # V_vertical_ellipsoidal(D, a, h)
        V = fv.V_vertical_ellipsoidal(__take_array_or_float(D), __take_array_or_float(a), __take_array_or_float(h))
    else:
        raise ValueError(f"`{orientation=}` not recognized. Must be one of [`Horizontal`, `Vertical`]")

    # if none of the inputs are time series we return a single result
    if all(isinstance(x, float) for x in [D, L, a, h]):
        # depending on the backend the calculated value can be a float or a scalar array
        return V if isinstance(V, float) else V.item()

    # otherwise we create a time series with one of the input indexes
    V_series = __create_series_with_index_of_inputs([D, L, a, h])
    V_series.loc[:] = V
    return V_series


@check_types
def filled_volume_spherical_head_vessel(
    D: Union[pd.Series, float],
    L: Union[pd.Series, float],
    a: Union[pd.Series, float],
    h: Union[pd.Series, float],
    orientation: Literal["Horizontal", "Vertical"] = "Horizontal",
) -> Union[pd.Series, float]:
    r"""Vessel volume (Spherical).

    Calculates partially full volume of a vertical or horizontal vessel with spherical convex heads. For vertical
    vessels the bottom is considered as spherical, but no provision for the top of the vessel is made here.

    For Hemispherical Head, :math:`a = D/2`.

    Args:
        D: Internal diameter [m].
            Diameter of the main cylindrical section.
        L: Straight length [m].
            Length of the main cylindrical section.
        a: Inside dish depth (a) [m].
            Distance the spherical head extends on one side
        h: Fluid level [m].
            Height, as measured up to where the fluid ends.
        orientation: Vessel orientation.

    Returns:
        Union[pd.Series, float]: Volume [m³]
            Volume of liquid in the vessel.

    References:
        Caleb Bell (2016-2021). fluids: Fluid dynamics component of Chemical Engineering Design Library (ChEDL)
        https://github.com/CalebBell/fluids.
    """
    if fv is None:
        raise ImportError(FLUIDS_REQUIRED)

    D, L, a, h = auto_align([D, L, a, h])

    if orientation == "Horizontal":
        # V_horiz_spherical(D, L, a, h, headonly=False)
        V = fv.V_horiz_spherical(
            __take_array_or_float(D),
            __take_array_or_float(L),
            __take_array_or_float(a),
            __take_array_or_float(h),
            False,
        )
    elif orientation == "Vertical":
        # V_vertical_spherical(D, a, h)
        V = fv.V_vertical_spherical(__take_array_or_float(D), __take_array_or_float(a), __take_array_or_float(h))
    else:
        raise ValueError(f"`{orientation=}` not recognized. Must be one of [`Horizontal`, `Vertical`]")

    # if none of the inputs are time series we return a single result
    if all(isinstance(x, float) for x in [D, L, a, h]):
        # depending on the backend the calculated value can be a float or a scalar array
        return V if isinstance(V, float) else V.item()

    # otherwise we create a time series with one of the input indexes
    V_series = __create_series_with_index_of_inputs([D, L, a, h])
    V_series.loc[:] = V
    return V_series


@check_types
def filled_volume_torispherical_head_vessel(
    D: Union[pd.Series, float],
    L: Union[pd.Series, float],
    f: Union[pd.Series, float],
    k: Union[pd.Series, float],
    h: Union[pd.Series, float],
    orientation: Literal["Horizontal", "Vertical"] = "Horizontal",
) -> Union[pd.Series, float]:
    r"""Vessel volume (Torispherical).

    Calculates partially full volume of a vertical or horizontal vessel with torispherical convex heads. For vertical
    vessels the bottom is considered as torispherical, but no provision for the top of the vessel is made here.

    For torispherical vessel heads, the following `f` and `k` parameters are used in standards. The default is ASME F&D.
    - **2:1 semi-elliptical**: `f = 0.9`, `k = 0.17`
    - **ASME F&D**: `f = 1`, `k = 0.06`
    - **ASME 80/6**: `f = 0.8`, `k = 0.06`
    - **ASME 80/10 F&D**: `f = 0.8`, `k = 0.1`
    - **DIN 28011**: `f = 1`, `k = 0.1`
    - **DIN 28013**: `f = 0.8`, `k = 0.154`

    Args:
        D: Internal diameter [m].
            Diameter of the main cylindrical section.
        L: Straight length [m].
            Length of the main cylindrical section.
        f: Dish-radius parameter (f) [-].
        k: Knuckle-radius parameter (k) [-].
        h: Fluid level [m].
            Height, as measured up to where the fluid ends.
        orientation: Vessel orientation.

    Returns:
        Union[pd.Series, float]: Volume [m³]
            Volume of liquid in the vessel.

    References:
        Caleb Bell (2016-2021). fluids: Fluid dynamics component of Chemical Engineering Design Library (ChEDL)
        https://github.com/CalebBell/fluids.
    """
    if fv is None:
        raise ImportError(FLUIDS_REQUIRED)

    D, L, f, k, h = auto_align([D, L, f, k, h])

    if orientation == "Horizontal":
        # V_horiz_torispherical(D, L, f, k, h, headonly=False)
        V = fv.V_horiz_torispherical(
            __take_array_or_float(D),
            __take_array_or_float(L),
            __take_array_or_float(f),
            __take_array_or_float(k),
            __take_array_or_float(h),
            False,
        )
    elif orientation == "Vertical":
        # V_vertical_torispherical(D, f, k, h)
        V = fv.V_vertical_torispherical(
            __take_array_or_float(D), __take_array_or_float(f), __take_array_or_float(k), __take_array_or_float(h)
        )
    else:
        raise ValueError(f"`{orientation=}` not recognized. Must be one of [`Horizontal`, `Vertical`]")

    # if none of the inputs are time series we return a single result
    if all(isinstance(x, float) for x in [D, L, f, k, h]):
        # depending on the backend the calculated value can be a float or a scalar array
        return V if isinstance(V, float) else V.item()

    # otherwise we create a time series with one of the input indexes
    V_series = __create_series_with_index_of_inputs([D, L, f, k, h])
    V_series.loc[:] = V
    return V_series
