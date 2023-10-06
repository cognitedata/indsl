# Copyright 2021 Cognite AS
import numpy as np
import pytest

from indsl.detect.change_point_detector import ed_pelt


rng = np.random.default_rng(42)


@pytest.mark.parametrize(
    "data, min_distance, expected",
    [
        (np.array([1.0, 1.0]), 1, np.array([])),
        (np.array([1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0]), 1, np.array([4, 9])),
        (np.concatenate([rng.normal(1.5, 0.1, 10), rng.normal(0, 0.1, 10)]), 1, np.array([10])),
        (
            np.concatenate([rng.normal(0, 0.1, 5), rng.normal(1.5, 0.1, 5), rng.normal(0, 0.1, 10)]),
            1,
            np.array([5, 11]),
        ),
    ],
)
def test_ed_pelt(data, min_distance, expected):
    """Unit test for the ED PELT change point detector."""
    # call method
    change_points = ed_pelt(data=data, min_distance=min_distance)

    # assertions
    np.testing.assert_array_equal(change_points, expected)
