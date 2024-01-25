import inspect
import json
import re

from typing import Optional

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
):
    if toolbox:
        return PREFIX + "_" + "TOOLBOX" + "_" + toolbox.upper().replace(" ", "_")
    if return_name and function_name:
        return PREFIX + "_" + function_name.upper().replace(" ", "_") + "_" + RETURN
    if parameter and function_name and description:
        return (
            PREFIX
            + "_"
            + function_name.upper().replace(" ", "_")
            + "_"
            + parameter.upper().replace(" ", "_")
            + "_"
            + DESCRIPTION
        )
    if parameter and function_name:
        return PREFIX + "_" + function_name.upper().replace(" ", "_") + "_" + parameter.upper().replace(" ", "_")
    if function_name and description:
        return PREFIX + "_" + function_name.upper().replace(" ", "_") + "_" + DESCRIPTION
    if function_name:
        return PREFIX + "_" + function_name.upper().replace(" ", "_")
    return None


def _create_value(value: str):
    return value.replace("\n", " ").replace(".", "")


# def convert_to_rendering_format(docstring: str) -> str:
#     docstring = re.sub(" +", " ", docstring)
#     try:
#         return docstring_to_markdown.rst_to_markdown(docstring)
#     except Exception:
#         return docstring


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


# TODO: ADD MARKDOWN AND THINK ABOUT VERSIONING, OUTLIER V1 THAT IS DEPRECATED


def _docstring_to_json(module):
    output_dict = {}
    for _, module in inspect.getmembers(indsl):
        toolbox_name = getattr(module, TOOLBOX_NAME, None)
        if toolbox_name is not None:
            output_dict[_create_key(toolbox=toolbox_name)] = toolbox_name

        # extract doctring from each function
        functions_to_export = getattr(module, COGNITE, [])
        functions_map = inspect.getmembers(module, inspect.isfunction)
        for name, function in functions_map:
            if name in functions_to_export:
                docstring = str(function.__doc__) if function.__doc__ else ""
                parsed_docstring = docstring_parser.parse(docstring, docstring_parser.DocstringStyle.GOOGLE)
                short_description = parsed_docstring.short_description if parsed_docstring.short_description else ""
                output_dict[_create_key(function_name=name)] = _create_value(short_description)

                # long description
                if parsed_docstring.long_description:
                    output_dict[_create_key(function_name=name, description=True)] = _convert_to_rendering_format(
                        parsed_docstring.long_description
                    )

                # Parameter names and descriptions
                parameters = parsed_docstring.params if parsed_docstring.params else []
                for parameter in parameters:
                    description = parameter.description if parameter.description else ""
                    parameter_name, description = _parse_docstring_element_text(parameter.description)
                    if parameter_name:
                        output_dict[_create_key(function_name=name, parameter=parameter.arg_name)] = parameter_name
                    if description:
                        output_dict[
                            _create_key(function_name=name, parameter=parameter.arg_name, description=True)
                        ] = _convert_to_rendering_format(description)

                # Name of the return value
                if parsed_docstring.returns:
                    return_name = parsed_docstring.returns.description
                    if return_name:
                        # return_name, _ = _parse_docstring_element_text(return_name)
                        output_dict[
                            _create_key(function_name=name, return_name=return_name)
                        ] = _parse_docstring_element_text(return_name)[0]

    with open("toolboxes.json", "w") as f:
        json.dump(output_dict, f, indent=4)
