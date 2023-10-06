import inspect
import logging
import re
import typing

from dataclasses import dataclass

import docstring_parser
import docstring_to_markdown
import pandas as pd
import versioning

from typing_utils import unwrap_optional_type_safe


log = logging.getLogger(__name__)

INDSL_INPUT_TYPES = (
    pd.Series,
    typing.Union[pd.Series, float],
    inspect._empty,  # type: ignore
)  # Types that are considered node inputs instead of node parameters when inspecting InDSL functions.


@dataclass
class OperationVariable:
    """
    Base representation of parameters, inputs and outputs

    """

    name: str
    description: str
    type: typing.Optional[typing.Type]
    param: str = None  # type: ignore
    default_value: typing.Optional[typing.Any] = None
    expected_pattern: typing.Optional[str] = None

    def __post_init__(self):
        if self.default_value is inspect.Parameter.empty:
            self.default_value = None
        if self.type is inspect.Parameter.empty:
            self.type = None

        unwrapped_type = unwrap_optional_type_safe(self.type)
        if inspect.isclass(unwrapped_type):
            if issubclass(unwrapped_type, pd.Timestamp):
                self.expected_pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d\d(?:.\d{3})?Z?$"
            elif issubclass(unwrapped_type, pd.Timedelta):
                self.expected_pattern = r"^\d+\s?(s|m|h|d)$"


class OperationVersion:
    """A single version of an operation"""

    def __init__(self, function, version):
        self.function = function
        self.version = version

        if versioning.is_versioned(function):
            assert versioning.get_version(function) == version
            self.deprecated = versioning.is_deprecated(function)
            self.changelog = versioning.get_changelog(function)
        else:
            self.deprecated = False
            self.changelog = None

        self.docstring_info = docstring_parser.parse(self.function.__doc__, docstring_parser.DocstringStyle.GOOGLE)
        self.docstring_params = {param.arg_name: param for param in self.docstring_info.params}
        for param in self.docstring_params.values():
            if param.description:
                param.description = self.convert_to_rendering_format(param.description)
        self.docstring_outputs = {
            output.return_name or f"output {i}": output for i, output in enumerate(self.docstring_info.many_returns)
        }
        sig = inspect.signature(self.function)
        self.arguments = sig.parameters
        # TODO Check for multiple return values if annotation is a tuple
        self.return_annotations = [sig.return_annotation or pd.Series]
        self.description = (
            self.convert_to_rendering_format(self.docstring_info.long_description)
            if self.docstring_info.long_description
            else None
        )
        self.name = self.docstring_info.short_description
        if self.name is not None:
            self.name = self.name.rstrip(".")
        self.outputs = self._make_output_info()
        self.inputs = []
        self.parameters = []
        for param in self.arguments.values():
            param_info = self._make_param_info(param)

            if param.annotation in INDSL_INPUT_TYPES:
                self.inputs.append(param_info)
            else:
                self.parameters.append(param_info)
        del self.arguments

    def _make_param_info(self, param: inspect.Parameter):
        name, description = self._parse_docstring_element_text(self.docstring_params, param.name)
        return OperationVariable(
            name=name, description=description, type=param.annotation, param=param.name, default_value=param.default
        )

    def _make_output_info(self):
        rv = []
        for i, type_ in enumerate(self.return_annotations):
            param_name = f"output {i}"
            name, description = self._parse_docstring_element_text(
                self.docstring_outputs, param_name, fallback_name=param_name
            )
            rv.append(OperationVariable(name=name, description=description, type=type_))
        return rv

    def _parse_docstring_element_text(self, docstring_container, item, fallback_name=None):
        """
        Retrieve name and description of an argument or return value based on it's docstring text

        Args:
            docstring_container: Where to look up the item (different for arguments and return values)
            item: Argument name or return value index
            fallback_name: Name to use in case of parsing failure

        Returns:
            name: Name of argument or return value
            description: Description text
        """
        if fallback_name is None:
            fallback_name = str(item)
        try:
            docstring_info = docstring_container[item]
            lines = docstring_info.description.splitlines()
            name = lines[0]
            description = "\n".join(lines[1:]) or None
        except LookupError:
            msg = f"Docstring seems invalid for {self.function} {fallback_name}"
            log.error(msg)

            name = fallback_name
            description = "<Unable to parse the description>"
        name = name.rstrip(".")
        return name, description

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    @staticmethod
    def convert_to_rendering_format(docstring: str) -> str:
        docstring = re.sub(" +", " ", docstring)
        try:
            return docstring_to_markdown.rst_to_markdown(docstring)
        except Exception as e:
            msg = f"Could not convert docstring to markdown: {docstring}. Exception: {e}"
            log.error(msg)
            return docstring


class Operation:
    """All versions of an operation"""

    def __init__(self, base_function: typing.Callable, category: str):
        self.category = category
        self._version_op_map = {}

        if versioning.is_versioned(base_function):
            # Collect all versions of the inDSL function
            self.op = versioning.get_name(base_function)
            for version in versioning.get_versions(self.op):
                func = versioning.get(self.op, version)
                self._version_op_map[version] = OperationVersion(func, version)
        else:
            # Unversioned inDSL functions get a default op_code and version
            self.op = base_function.__name__
            self._version_op_map["1.0"] = OperationVersion(base_function, "1.0")

    @property
    def versions(self):
        """Return (ordered) list of all versions of the operation"""
        return list(self._version_op_map.values())

    def get_latest_version(self):
        """Return latest version of the operation"""
        return self.versions[-1]

    def get_version(self, version: str):
        """Return specific version of the operation"""
        return self._version_op_map[version]
