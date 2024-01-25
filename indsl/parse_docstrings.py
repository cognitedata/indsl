import inspect
import json
import re

from typing import Optional
import typing
from indsl import versioning


import docstring_parser

from docstring_to_markdown.rst import rst_to_markdown

import indsl


PREFIX = "INDSL"
PARAMETER = "PARAMETER"
DESCRIPTION = "DESCRIPTION"
RETURN = "RETURN"
TOOLBOX_NAME = "TOOLBOX_NAME"
COGNITE = "__cognite__"


# def _parse_docstring_element_textcreate_key(*args):
#     return "_".join(arg.upper().replace(" ", "_") for arg in args)


def _create_key(
    toolbox: Optional[str] = None,
    function_name: Optional[str] = None,
    description: Optional[bool] = False,
    parameter: Optional[str] = None,
    return_name: Optional[str] = None,
    version: Optional[str] = "",
):
    if version:
        version = "_" + version
    else:
        version = ""
    if toolbox:
        return PREFIX + "_" + "TOOLBOX" + "_" + toolbox.upper().replace(" ", "_")
    if return_name and function_name:
        return PREFIX + "_" + function_name.upper().replace(" ", "_") + "_" + RETURN + version
    if parameter and function_name and description:
        return (
            PREFIX
            + "_"
            + function_name.upper().replace(" ", "_")
            + "_"
            + parameter.upper().replace(" ", "_")
            + "_"
            + DESCRIPTION
            + version
        )
    if parameter and function_name:
        return (
            PREFIX + "_" + function_name.upper().replace(" ", "_") + "_" + parameter.upper().replace(" ", "_") + version
        )
    if function_name and description:
        return PREFIX + "_" + function_name.upper().replace(" ", "_") + "_" + DESCRIPTION + version
    if function_name:
        return PREFIX + "_" + function_name.upper().replace(" ", "_") + version
    return None


def _parse_docstring_element_text(docstring):
    lines = docstring.splitlines()
    name = lines[0]
    description = "\n".join(lines[1:]) or None
    name = name.rstrip(".")

    return name, description


def _convert_to_rendering_format(docstring: str) -> str:
    docstring = re.sub(" +", " ", docstring)
    try:
        return rst_to_markdown(docstring)
    except Exception:
        return docstring


def _generate_parameters(name: str, output_dict: dict, parameters, version: Optional[str] = None):
    for parameter in parameters:
        description = parameter.description if parameter.description else ""
        parameter_name, description = _parse_docstring_element_text(parameter.description)
        if parameter_name:
            output_dict[_create_key(function_name=name, parameter=parameter.arg_name, version=version)] = parameter_name
        if description:
            output_dict[
                _create_key(function_name=name, parameter=parameter.arg_name, description=True, version=version)
            ] = _convert_to_rendering_format(description)


# TODO: REFACTOR


def _generate_key_for_function(function: typing.Callable, name: str, output_dict: dict, version: Optional[str] = None):
    docstring = str(function.__doc__) if function.__doc__ else ""
    parsed_docstring = docstring_parser.parse(docstring, docstring_parser.DocstringStyle.GOOGLE)
    short_description = parsed_docstring.short_description if parsed_docstring.short_description else ""
    output_dict[_create_key(function_name=name, version=version)] = short_description

    # long description
    if parsed_docstring.long_description:
        output_dict[_create_key(function_name=name, description=True, version=version)] = _convert_to_rendering_format(
            parsed_docstring.long_description
        )

    # Parameter names and descriptions
    parameters = parsed_docstring.params if parsed_docstring.params else []
    _generate_parameters(name, output_dict, parameters, version=version)

    # Name of the return value
    if parsed_docstring.returns:
        return_name = parsed_docstring.returns.description
        if return_name:
            # return_name, _ = _parse_docstring_element_text(return_name)
            output_dict[
                _create_key(function_name=name, return_name=return_name, version=version)
            ] = _parse_docstring_element_text(return_name)[0]


def _docstring_to_json(module):
    output_dict = {}
    for _, module in inspect.getmembers(indsl, inspect.ismodule):
        toolbox_name = getattr(module, TOOLBOX_NAME, None)
        if toolbox_name is not None:
            output_dict[_create_key(toolbox=toolbox_name)] = toolbox_name
        print(toolbox_name)

        # extract doctring from each function
        functions_to_export = getattr(module, COGNITE, [])
        functions_map = inspect.getmembers(module, inspect.isfunction)
        for name, function in functions_map:
            if name in functions_to_export:
                if versioning.is_versioned(function):
                    # Collect all versions of the inDSL function
                    function_name = versioning.get_name(function)
                    versions = versioning.get_versions(function_name)
                    for version in versions:
                        func = versioning.get(function_name, version)
                        if version == versions[-1]:
                            _generate_key_for_function(function, name, output_dict=output_dict)
                        else:
                            _generate_key_for_function(func, function_name, output_dict=output_dict, version=version)

                else:
                    # Unversioned inDSL functions get a default op_code and version
                    _generate_key_for_function(function, name, output_dict=output_dict)

    with open("toolboxes.json", "w") as f:
        json.dump(output_dict, f, indent=4)
