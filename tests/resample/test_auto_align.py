# Copyright 2021 Cognite AS
import numpy as np

from indsl.resample.auto_align import auto_reindex
from indsl.ts_utils import add

from ..generate_data import create_uniform_data


def test_auto_reindex():
    d1 = create_uniform_data(np.ones(3), frequency="5min")
    d2 = create_uniform_data(np.ones(3), frequency="3min")

    r1, r2 = auto_reindex([d1, d2])
    assert r1.index.equals(r2.index)
    assert len(r1) == 4

    r1, r2 = auto_reindex([d1, d2], enabled=False)
    assert r1.index.equals(d1.index)
    assert r2.index.equals(d2.index)

    r1, r2 = auto_reindex([d1, d1])
    assert r1.index.equals(d1.index)
    assert r2.index.equals(d1.index)
    assert len(r1) == len(d1)


def test_auto_realign_in_add_function():
    a = create_uniform_data(np.ones(3), frequency="5min")
    b = create_uniform_data(np.ones(3), frequency="3min")

    # Without auto-alignment we only get Nan
    r = add(a, b, align_timesteps=False)
    assert np.isnan(r[1:]).all()
    r = add(a, b)  # Default is auto_align disabled
    assert np.isnan(r[1:]).all()

    # With auto-realignment, no values are Nan
    r = add(a, a, align_timesteps=True)
    assert np.isfinite(r).all()
    r = add(a, b, align_timesteps=True)
    assert np.isfinite(r).all()

    # Test that auto-alignment also handles the case where one of arguments is not a time-series
    r = add(a, 1, align_timesteps=True)
    assert (r == a + 1).all()
