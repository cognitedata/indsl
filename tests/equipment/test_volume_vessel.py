import pandas as pd
import pytest

from numpy.testing import assert_allclose
from pandas.testing import assert_series_equal

from indsl.equipment.volume_vessel import (
    filled_volume_ellipsoidal_head_vessel,
    filled_volume_spherical_head_vessel,
    filled_volume_torispherical_head_vessel,
)


# reusable examples
time_index = pd.date_range(start="2022-07-01 00:00:00", end="2022-07-01 01:00:00", periods=8)
level_sensor_horizontal = pd.Series(index=time_index, data=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
level_sensor_vertical = pd.Series(index=time_index, data=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])


@pytest.mark.extras
def test_horizontal_ellipsoidal_vessel_scalar():
    # all scalar
    Vh = filled_volume_ellipsoidal_head_vessel(D=3.0, L=6.0, a=0.8, h=0.5, orientation="Horizontal")
    expected_Vh = 5.204728480805875
    assert_allclose(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_horizontal_ellipsoidal_vessel_vector():
    # default case -> level is time series
    Vh = filled_volume_ellipsoidal_head_vessel(D=3.0, L=6.0, a=0.8, h=level_sensor_horizontal, orientation="Horizontal")
    expected_Vh = pd.Series(
        data=[
            0.0,
            1.310327493044075,
            3.7280624791679524,
            6.822626969357778,
            10.401866806934775,
            14.33008020921482,
            18.49598703260846,
            22.80056320660493,
        ],
        index=Vh.index,
        name="Volume",
        dtype="float64",
    )
    assert_series_equal(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_vertical_ellipsoidal_vessel_scalar():
    # all scalar
    Vh = filled_volume_ellipsoidal_head_vessel(D=3.0, L=6.0, a=0.8, h=3.5, orientation="Vertical")
    expected_Vh = 22.855086554865746
    assert_allclose(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_vertical_ellipsoidal_vessel_vector():
    # default case -> level is time series
    Vh = filled_volume_ellipsoidal_head_vessel(D=3.0, L=6.0, a=0.8, h=level_sensor_vertical, orientation="Vertical")
    expected_Vh = pd.Series(
        data=[
            0.0,
            1.7487380981896312,
            5.183627878423159,
            8.717919613711675,
            12.252211349000193,
            15.78650308428871,
            19.320794819577227,
            22.855086554865746,
        ],
        index=Vh.index,
        name="Volume",
        dtype="float64",
    )
    assert_series_equal(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_horizontal_spherical_vessel_scalar():
    # all scalar
    Vh = filled_volume_spherical_head_vessel(D=3.0, L=6.0, a=0.8, h=0.5, orientation="Horizontal")
    expected_Vh = 4.996023436633497
    assert_allclose(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_horizontal_spherical_vessel_vector():
    # default case -> level is time series
    Vh = filled_volume_spherical_head_vessel(D=3.0, L=6.0, a=0.8, h=level_sensor_horizontal, orientation="Horizontal")
    expected_Vh = pd.Series(
        data=[
            0.0,
            1.2569508729793075,
            3.574318120623258,
            6.5589072511381925,
            10.032537922976188,
            13.864188429032456,
            17.94249254486732,
            22.165726700918874,
        ],
        index=Vh.index,
        name="Volume",
        dtype="float64",
    )
    assert_series_equal(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_vertical_spherical_vessel_scalar():
    # all scalar
    Vh = filled_volume_spherical_head_vessel(D=3.0, L=6.0, a=0.8, h=3.5, orientation="Vertical")
    expected_Vh = 22.180691331895137
    assert_allclose(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_vertical_spherical_vessel_vector():
    # default case -> level is time series
    Vh = filled_volume_spherical_head_vessel(D=3.0, L=6.0, a=0.8, h=level_sensor_vertical, orientation="Vertical")
    expected_Vh = pd.Series(
        data=[
            0.0,
            1.2877257387370662,
            4.50923265545255,
            8.043524390741068,
            11.577816126029584,
            15.112107861318101,
            18.646399596606617,
            22.180691331895137,
        ],
        index=Vh.index,
        name="Volume",
        dtype="float64",
    )
    assert_series_equal(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_horizontal_torispherical_vessel_scalar():
    # all scalar
    Vh = filled_volume_torispherical_head_vessel(D=3.0, L=6.0, f=1.0, k=0.1, h=0.5, orientation="Horizontal")
    expected_Vh = 5.049720333662088
    assert_allclose(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_horizontal_torispherical_vessel_vector():
    # default case -> level is time series
    Vh = filled_volume_torispherical_head_vessel(
        D=3.0, L=6.0, f=1.0, k=0.1, h=level_sensor_horizontal, orientation="Horizontal"
    )
    expected_Vh = pd.Series(
        data=[
            0.0,
            1.2890408529260575,
            3.6304497813863406,
            6.599639079718773,
            10.017250658068436,
            13.758282933657442,
            17.72014502463491,
            21.811097923505415,
        ],
        index=Vh.index,
        name="Volume",
        dtype="float64",
    )
    assert_series_equal(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_vertical_torispherical_vessel_scalar():
    # all scalar
    Vh = filled_volume_torispherical_head_vessel(D=3.0, L=6.0, f=1.0, k=0.1, h=3.5, orientation="Vertical")
    expected_Vh = 23.30299616002319
    assert_allclose(Vh, expected_Vh, atol=1e-6)


@pytest.mark.extras
def test_vertical_torispherical_vessel_vector():
    # default case -> level is time series
    Vh = filled_volume_torispherical_head_vessel(
        D=3.0, L=6.0, f=1.0, k=0.1, h=level_sensor_vertical, orientation="Vertical"
    )
    expected_Vh = pd.Series(
        data=[
            0.0,
            2.1000872614775377,
            5.631537483580606,
            9.165829218869122,
            12.70012095415764,
            16.23441268944616,
            19.768704424734675,
            23.30299616002319,
        ],
        index=Vh.index,
        name="Volume",
        dtype="float64",
    )
    assert_series_equal(Vh, expected_Vh, atol=1e-6)
