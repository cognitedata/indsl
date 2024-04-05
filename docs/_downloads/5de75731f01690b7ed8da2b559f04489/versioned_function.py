# Copyright 2021 Cognite AS
"""
Function versioning
===================

InDSL comes with the :py:mod:`indsl.versioning` module, which allows to implement multiple versions of InDSL functions.
As a library user, one can then select and execute a specific function version.

Example
-------

In this example, we implement the `abs_diff` function, which computes the element-wise absolute difference of two time-series.
We will first implement a naive version of that function, which we name version 1.0 (versioning in inDSL always starts with 1.0),
followed by a more robust version 1.1.

"""


# %%
# Implementation
# --------------
#
# Implementation of v1.0
# ^^^^^^^^^^^^^^^^^^^^^^
#
# We begin with a simple implementation:

import pandas as pd

from indsl import versioning


@versioning.register(version="1.0", deprecated=True)
def abs_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).abs()


# %%
# They key in this function definition is the :func:`indsl.versioning.register` decorator.
# This decorator registers the function as a versioned function with name `abs_diff` and version `1.0`.
# We also mark the function as deprecated, since we will soon implement a new version of the function.
# This means that we can retrieve and execute this version, even after newer version of the same functions have been registered.


# %%
# Our, initial implementation is not very robust and results easily in `nan` outputs.
# This happens specifically when we apply `abs`diff` to time-series with non-matching indices:

idx = pd.date_range("2022-01-01", periods=5, freq="1h")
a = pd.Series([1, 2, 3, 4, 5], index=idx)

idx = pd.date_range("2022-01-01", periods=3, freq="2h")
b = pd.Series([1, 3, 5], index=idx)

abs_diff(a, b)

# %%
# Version 1.1 will fix this issue through a more robust implementation.

# %%
# Implementation of v1.1
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Next, we implement the new version of the `abs_diff` and mark it as version 1.1.
#

from indsl.resample import reindex  # noqa


@versioning.register(version="1.1")  # type: ignore
def abs_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    a, b = reindex(a, b)
    return (a - b).abs()


# %%
# We rely on the build-in function `reindex` to align the indices of the time-series (using linear-interpolation) before performing the operations.

abs_diff(a, b)

# %%
# Getting versioned functions and their versions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# We can get a list of all versioned functions with:

versioning.get_registered_functions()

# %%
# We can retrieve which versions we have of a function with:

versioning.get_versions("abs_diff")

# %%
# Running versioned functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
#
# We can access and run specific function version with the `versioning.get` command:
abs_diff_v1 = versioning.get("abs_diff", version="1.0")
abs_diff_v1(a, b)

# %%
# Omitting the version argument will automatically select the latest version
abs_diff_v1_1 = versioning.get("abs_diff")
abs_diff_v1_1(a, b)

# sphinx_gallery_thumbnail_path = '_static/images/versioning_thumbnail.png'
