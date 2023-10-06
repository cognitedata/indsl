# Copyright 2023 Cognite AS
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Sequence

from packaging.version import Version

from .type_check import check_types


_registered_funcs: DefaultDict[str, Dict[str, Any]] = defaultdict(dict)


@check_types
def register(
    version: str,
    name: Optional[str] = None,
    deprecated: Optional[bool] = False,
    changelog: Optional[str] = None,
) -> Callable[..., Any]:
    """Decorator to register a new versioned function.

    Args:
        version: Version number of the function
            The version number follows the standard version scheme for Python packages
            as defined in `PEP 440 <https://packaging.pypa.io/en/latest/version.html>`_.

            The following are valid version numbers (shown in the order that
            would be obtained by sorting according to the supplied cmp function):

                0.4       0.4.0  (these two are equivalent)
                0.4.1
                0.5a1
                0.5b3
                0.5
                0.9.6
                1.0
                1.0.4a3
                1.0.4b1
                1.0.4

            The following are examples of invalid version numbers:

                1.3pl1
                a
        name: Name of the function
            The name under which the function should be registered. If no name
            is specified, the name of the function is used.
        deprecated: Flag if function is deprecated. Default: False
        changelog: Description of what has changed since the previous version. Default: None
    """
    # Test that the version numbering is valid
    Version(version)

    def register_decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Register a new versioned function."""
        key = name or func.__name__

        if version in _registered_funcs[key]:
            raise ValueError(f"Function {key} with version {version} is already registered")

        _registered_funcs[key][version] = func

        setattr(func, "__versioning_version__", version)
        setattr(func, "__versioning_name__", key)
        setattr(func, "__versioning_deprecated__", deprecated)
        setattr(func, "__versioning_changelog__", changelog)

        return func

    return register_decorator


def get_registered_functions() -> List[str]:
    """Return the list of registered function names."""
    return list(_registered_funcs.keys())


@check_types
def get_versions(name: str) -> List[str]:
    """Return the list of available versions of a function name.

    Sorted from low to highest version
    """
    versions = _registered_funcs[name].keys()
    return sorted(versions, key=Version)


@check_types
def get_name(func: Callable[..., Any]) -> str:
    """Return the name with which the function was registered."""
    try:
        return getattr(func, "__versioning_name__")
    except AttributeError:
        raise ValueError("Cannot get name of an un-registerd function")


@check_types
def get_version(func: Callable[..., Any]) -> Optional[str]:
    """Return the version of a function.

    If the function is not registered, None is returned
    """
    return getattr(func, "__versioning_version__", None)


@check_types
def is_versioned(func: Callable[..., Any]) -> bool:
    """Return true if the version is versioned."""
    return hasattr(func, "__versioning_version__")


@check_types
def is_deprecated(func: Callable[..., Any]) -> Optional[bool]:
    """Return true if the version is deprecated.

    If the function is not registered, None is returned
    """
    return getattr(func, "__versioning_deprecated__", None)


@check_types
def get_changelog(func: Callable[..., Any]) -> Optional[str]:
    """Return changelog of this version.

    If the function is not registered, None is returned
    """
    return getattr(func, "__versioning_changelog__", None)


@check_types
def get(name: str, version: Optional[str] = None) -> Callable[..., Any]:
    """Return one of the versions of a function.

    If version is None, the latest version is returned
    """
    if version is None:
        version = get_versions(name)[-1]

    if name not in _registered_funcs:
        raise ValueError(f"Function {name} is not registered")
    if version not in _registered_funcs[name]:
        raise ValueError(f"Version {version} for function {name} is not registered")
    return _registered_funcs[name][version]


@check_types
def run(
    name: str, version: Optional[str] = None, args: Optional[Sequence] = None, kwargs: Optional[dict] = None
) -> Any:
    """Run a version of a function.

    If version is None, the latest version is executed
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    return get(name, version)(*args, **kwargs)
