# Copyright 2021 Cognite AS
import inspect

import docstring_parser
import pandas as pd
import pytest

import indsl


def _get_all_operations():
    indsl_functions = []
    for _, module in inspect.getmembers(indsl):
        toolbox_name = getattr(module, "TOOLBOX_NAME", None)
        if toolbox_name is None:
            continue
        functions_to_export = getattr(module, "__cognite__", [])
        functions_map = inspect.getmembers(module, inspect.isfunction)
        for name, function in functions_map:
            if name in functions_to_export:
                indsl_functions.append(function)
    return indsl_functions


def _parse_docstring_element_text(docstring_container, item):
    docstring_info = docstring_container[item]
    return docstring_info.description.splitlines()


indsl_functions = _get_all_operations()


def get_unwrapped_file(function):
    """Return the filename of the un-decorated function"""
    if hasattr(function, "__wrapped__"):
        return get_unwrapped_file(function.__wrapped__)
    else:
        return inspect.getfile(function)


@pytest.mark.parametrize("function", indsl_functions)
def test_docstrings_are_valid(function):
    sig = inspect.signature(function)
    docstring_info = docstring_parser.parse(function.__doc__, docstring_parser.DocstringStyle.GOOGLE)
    docstring_params = {param.arg_name: param for param in docstring_info.params}
    docstring_outputs = {
        output.return_name or f"output {i}": output for i, output in enumerate(docstring_info.many_returns)
    }
    return_annotations = [sig.return_annotation or pd.Series]

    err_msg = f"Invalid docstring for function '{function.__name__}' in {get_unwrapped_file(function)}"

    assert function.__doc__ is not None, f"{err_msg} Missing docstring."

    assert (
        len_short_desc := len(docstring_info.short_description)
    ) <= 31, f"{err_msg} \
            The short description should not have more than 30 characters, has {len_short_desc} characters."

    max_len_short_desc = 30
    for param in sig.parameters.values():
        lines = _parse_docstring_element_text(docstring_params, param.name)
        assert len(lines) > 0, f"{err_msg} Missing docstring for parameter {param.name}."

        # Ignore LaTeX units in formulas notation in the parameter name length
        # Parameter name units MUST be:
        # * Enclosed in square brackets ``[]``
        # * In Roman (not italic) font
        # * If using LaTeX language, use the ``:math:`` inline formula command, and the command ``\mathrm{}`` to render
        #   the unis in Roman font.
        # * Placed at the end of the string

        # Check that units are within brackets and placed at the end of the parameter name string, before the
        # punctuation mark.
        if "]" in lines[0]:
            assert lines[0].find("].") == len(lines[0]) - 2, (
                f"{err_msg} Units in parameter names must be enclosed in square brackets, placed at the end of the "
                f"string, and followed by punctuation mark. "
                f'Instead got: "{lines[0]}"'
            )
            assert lines[0].find("[") != -1, (
                f"{err_msg} Units in parameter names must be enclosed in square brackets, seems like you missed the "
                f'opening bracket "[". We got: "{lines[0]}"'
            )
        if "[" in lines[0]:
            assert lines[0].find("]") > lines[0].find("["), (
                f"{err_msg} Units in parameter names must be enclosed in square brackets, seems like you missed the "
                f'closing bracket "]". We got: "{lines[0]}"'
            )

        # Check that LaTeX typed units are in Roman font
        if "[:math:" in lines[0]:
            assert lines[0].find("mathrm") != -1, (
                f"{err_msg} Units in parameter names must be typed using Roman font. Seems like you forgot to use the "
                f'command "\\"mathrm" with the LaTeX mathematical notation.'
            )

        # Ignore characters representing units ... Anything inside square brackets.
        # Also accounts for the space between the name and the brackets and the punctuation at the end.
        len_lines = _ignore_latex_math_within_sqbrackets(lines)

        assert (len_short_desc := len_lines) <= max_len_short_desc, (
            f"{err_msg} The short description for parameter {param.name}, excluding units, should not have "
            f"more than {max_len_short_desc} characters. It has {len_short_desc} characters."
        )
        assert (
            len(diff := (set(docstring_params.keys()) - set(sig.parameters.keys()))) == 0
        ), f"{err_msg} Docstring describes non-existent parameter {diff}"

    for i in range(len(return_annotations)):
        lines = _parse_docstring_element_text(docstring_outputs, f"output {i}")
        assert len(lines) > 0, f"{err_msg} Missing docstring for output {param.name}."

        # Ignore characters representing units ... Anything inside square brackets.
        # Also accounts for the space between the name and the brackets and the punctuation at the end.
        len_lines = _ignore_latex_math_within_sqbrackets(lines)

        assert (len_short_desc := len_lines) <= max_len_short_desc, (
            f"{err_msg} The short description for output {i}, excluding units, should not have more than "
            f"{max_len_short_desc} characters, has {len_short_desc} characters."
        )


def _ignore_latex_math_within_sqbrackets(lines):
    if "[" in lines[0]:
        # -2 at the end of the formula below is to ignore the space between the parameter name and units, and the
        # punctuation mark at the end, in the character count.
        len_lines = len(lines[0]) - (lines[0].find("]") - lines[0].find("[") - 1) - 2
    else:
        len_lines = len(lines[0])
    return len_lines
