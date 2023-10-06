# Copyright 2021 Cognite AS
import pandas as pd
import pandas.testing as tm
import pytest

from numpy.random import MT19937, RandomState, SeedSequence

from indsl.not_listed_operations import no_op


@pytest.mark.core
def test_basic_stats_functionality():
    rs = RandomState(MT19937(SeedSequence(1975)))
    inp_series = pd.Series(rs.random(20))
    res = no_op(inp_series)

    tm.assert_series_equal(res, inp_series)
