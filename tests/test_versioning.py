# Copyright 2022 Cognite AS
import inspect

import pytest

from indsl import versioning


@versioning.register(version="2.0b1", deprecated=True, changelog="Added debug flag")
def add_PYTEST(a, b, debug=True):
    if debug:
        print("Debug is on")
    return a + b


@versioning.register(version="1.0", deprecated=True)  # type: ignore
def add_PYTEST(a, b):  # noqa: F811
    return a + b + 1


# We can register functions under a different name
@versioning.register(version="1.2", name="add_PYTEST")
def add(a, b):
    return a + b


def divide(a, b):
    return a / b


def test_get_registered_functions():
    assert "add_PYTEST" in versioning.get_registered_functions()


def test_max_one_non_deprecated_function():
    func_names = versioning.get_registered_functions()
    for func_name in func_names:
        versions = versioning.get_versions(func_name)
        funcs = [versioning.get(func_name, v) for v in versions]
        non_deprecated_versions = [v for v, f in zip(versions, funcs) if not versioning.is_deprecated(f)]

        assert (
            len(non_deprecated_versions) <= 1
        ), f"Expected at most one non-deprecated version for function '{func_name}', got versions {non_deprecated_versions}"


def test_get_versions():
    assert versioning.get_versions("add_PYTEST") == ["1.0", "1.2", "2.0b1"]


def test_get_function():
    add_new = versioning.get("add_PYTEST", "2.0b1")
    assert add_new(1, 2) == 3

    add_latest = versioning.get("add_PYTEST")
    assert add_latest(1, 2, False) == 3

    add_old = versioning.get("add_PYTEST", "1.0")
    assert add_old(1, 2) == 4


def test_run_function():
    assert versioning.run("add_PYTEST", "2.0b1", args=(1, 2, False)) == 3
    assert versioning.run("add_PYTEST", "2.0b1", args=(1, 2)) == 3
    assert versioning.run("add_PYTEST", "1.0", args=(1, 2)) == 4
    assert versioning.run("add_PYTEST", kwargs={"a": 1, "b": 2}) == 3


def test_direct_exection():
    assert add_PYTEST(1, 2) == 4


def test_cannot_run_non_registered_functions():
    with pytest.raises(ValueError):
        versioning.run("sub", "2.0")


def test_cannot_run_non_registered_versions():
    with pytest.raises(ValueError):
        versioning.run("add_PYTEST", "2.0")


def test_cannot_register_same_version_twice():
    with pytest.raises(ValueError):

        @versioning.register("1.0")
        def add_PYTEST(a, b):
            return a + b


def test_is_versioned():
    assert versioning.is_versioned(add)
    assert not versioning.is_versioned(divide)


def test_is_deprecated():
    add_old = versioning.get("add_PYTEST", "1.0")
    assert versioning.is_deprecated(add_old)

    add_new = versioning.get("add_PYTEST", "1.2")
    assert not versioning.is_deprecated(add_new)

    add_newest = versioning.get("add_PYTEST", "2.0b1")
    assert versioning.is_deprecated(add_newest)


def test_changelog():
    add_old = versioning.get("add_PYTEST", "1.0")
    assert versioning.get_changelog(add_old) is None

    add_new = versioning.get("add_PYTEST", "1.2")
    assert versioning.get_changelog(add_new) is None

    add_newest = versioning.get("add_PYTEST", "2.0b1")
    assert versioning.get_changelog(add_newest) == "Added debug flag"


def test_get_version_from_function():
    assert versioning.get_version(add) == "1.2"


def test_get_version_from_unregistered_function_returns_None():
    assert versioning.get_version(divide) is None


def test_get_name_from_function():
    assert versioning.get_name(add) == "add_PYTEST"


def test_cannot_get_name_from_unregistered_function():
    with pytest.raises(ValueError):
        versioning.get_name(divide)


def test_all_registered_functions_start_with_version_1_0():
    for func_name in versioning.get_registered_functions():
        v = versioning.get_versions(func_name)
        assert "1.0" == v[0], f"Expected {func_name} with version '1.0', but got {v}"


def test_non_listed_functions_with_old_naming_scheme_are_registered():
    assert versioning.get_versions("WAVELET_FILTER") == ["1.0"]

    wavelet_filter_old_naming_scheme = versioning.get("WAVELET_FILTER", "1.0")
    wavelet_filter_new_naming_scheme = versioning.get("wavelet_filter", "1.0")

    assert inspect.signature(wavelet_filter_new_naming_scheme) == inspect.signature(
        wavelet_filter_old_naming_scheme
    ), "Expected function WAVELET_FILTER v1.0 to be the same as wavelet_filter v1.0"
