import inspect
import json
import re
import typing

from typing import Optional

import docstring_parser

from docstring_to_markdown.rst import rst_to_markdown

import indsl

from indsl import versioning


PREFIX = "INDSL"
PARAMETER = "PARAMETER"
DESCRIPTION = "DESCRIPTION"
RETURN = "RETURN"
TOOLBOX_NAME = "TOOLBOX_NAME"
TOOLBOX = "TOOLBOX"
COGNITE = "__cognite__"


# Format the string to uppercase and replace spaces with underscores
def _format_str(s: Optional[str]) -> str:
    return s.upper().replace(" ", "_") if s else ""


# Create keys for the JSON file
def _create_key(
    toolbox: Optional[str] = None,
    function_name: Optional[str] = None,
    description: Optional[bool] = False,
    parameter: Optional[str] = None,
    return_name: Optional[str] = None,
    version: Optional[str] = "",
) -> Optional[str]:
    elements = [PREFIX]

    if toolbox:
        elements.append(TOOLBOX)
        elements.append(_format_str(toolbox))
    else:
        if function_name:
            elements.append(_format_str(function_name))
            if return_name:
                elements.append(RETURN)
            elif parameter:
                elements.append(_format_str(parameter))
                if description:
                    elements.append(DESCRIPTION)
            elif description:
                elements.append(DESCRIPTION)

    version_str = f"_{_format_str(version)}" if version else ""
    return "_".join(elements) + version_str if elements else None


# Parse the docstring element text
def _parse_docstring_element_text(docstring):
    lines = docstring.splitlines()
    name = lines[0]
    description = "\n".join(lines[1:]) or None
    name = name.rstrip(".")

    return name, description


# Convert the docstring to rendering format for markdown
def _convert_to_rendering_format(docstring: str) -> str:
    docstring = re.sub(" +", " ", docstring)
    try:
        return rst_to_markdown(docstring)
    except Exception:
        return docstring


# Generate parameters
def _generate_parameters(name: str, output_dict: dict, parameters, version: Optional[str] = None):
    for parameter in parameters:
        param_name, description = _parse_docstring_element_text(parameter.description)
        if param_name:
            key = _create_key(function_name=name, parameter=parameter.arg_name, version=version)
            output_dict[key] = param_name
        if description:
            desc_key = _create_key(function_name=name, parameter=parameter.arg_name, description=True, version=version)
            output_dict[desc_key] = _convert_to_rendering_format(description)


# TODO: REFACTOR AND FIGURE OUT WHY TOOLBOX_NAME IS INCLUDED IN THE JSON FILE. FIGURE OUT WHY VERSIONING FUNCTION APPEARS IN THE JSON.


# Generation of the keys and values
def _generate_key_for_function(function: typing.Callable, name: str, output_dict: dict, version: Optional[str] = None):
    docstring = str(function.__doc__) if function.__doc__ else ""
    parsed_docstring = docstring_parser.parse(docstring, docstring_parser.DocstringStyle.GOOGLE)

    # Function name
    short_description = parsed_docstring.short_description or ""
    output_dict[_create_key(function_name=name, version=version)] = short_description

    # Function description
    if parsed_docstring.long_description:
        long_desc_key = _create_key(function_name=name, description=True, version=version)
        output_dict[long_desc_key] = _convert_to_rendering_format(parsed_docstring.long_description)

    # Function parameters
    if parsed_docstring.params:
        _generate_parameters(name, output_dict, parsed_docstring.params, version)

    # Function return
    if parsed_docstring.returns:
        return_name = _parse_docstring_element_text(parsed_docstring.returns.description)[0]
        if return_name:
            return_key = _create_key(function_name=name, return_name=return_name, version=version)
            output_dict[return_key] = return_name


# Add versioning if the function is versioned
def _generate_function_name_and_versioning(output_dict: dict, module):
    functions_to_export = getattr(module, COGNITE, [])
    functions_map = inspect.getmembers(module, inspect.isfunction)

    for name, function in filter(lambda f: f[0] in functions_to_export, functions_map):
        if versioning.is_versioned(function):
            _process_versioned_function(function, name, output_dict)
        else:
            _generate_key_for_function(function, name, output_dict=output_dict)


# Process versioned function by adding only the deprecated version to the keys.
# The latest version does not have a version number in the key.
def _process_versioned_function(function, name, output_dict):
    function_name = versioning.get_name(function)
    versions = versioning.get_versions(function_name)

    for version in versions:
        func = versioning.get(function_name, version)
        current_function = func if version != versions[-1] else function
        _generate_key_for_function(
            current_function,
            function_name if version != versions[-1] else name,
            output_dict=output_dict,
            version=version if version != versions[-1] else None,
        )


# Write the keys and values to the JSON file
def _docstring_to_json(module):
    output_dict = {}
    for _, module in inspect.getmembers(indsl, inspect.ismodule):
        toolbox_name = getattr(module, TOOLBOX_NAME, None)
        if toolbox_name is not None:
            output_dict[_create_key(toolbox=toolbox_name)] = toolbox_name

        # extract doctring from each function
        _generate_function_name_and_versioning(output_dict, module)

    with open("toolboxes.json", "w") as f:
        json.dump(output_dict, f, indent=4)
