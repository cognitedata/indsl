import inspect
import json
import re
import typing

from typing import Optional

import docstring_parser
import requests

from docstring_to_markdown.rst import rst_to_markdown

import indsl

from indsl import versioning
from indsl.create_translation_key import create_key


TOOLBOX_NAME = "TOOLBOX_NAME"
COGNITE = "__cognite__"

LOCIZE_API_KEY = "9e22ecf1-1fe7-41f2-827b-9f51b7c67f8b"
LOCIZE_PROJECT_ID = "52167e6e-aea8-4433-83f7-f65976dd5f18"
NAMESPACE = "vebjorn-translations"
OUTPUT_FILE = "keys_from_locize.json"


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
def _generate_key_for_parameters(name: str, output_dict: dict, parameters, version: Optional[str] = None):
    for parameter in parameters:
        param_name, description = _parse_docstring_element_text(parameter.description)
        if param_name:
            key = create_key(function_name=name, parameter=parameter.arg_name, version=version)
            output_dict[key] = param_name
        if description:
            description_key = create_key(
                function_name=name, parameter=parameter.arg_name, description=True, version=version
            )
            output_dict[description_key] = _convert_to_rendering_format(description)


# Generation of the keys and values
def _generate_key_for_function(function: typing.Callable, name: str, output_dict: dict, version: Optional[str] = None):
    docstring = str(function.__doc__) if function.__doc__ else ""
    parsed_docstring = docstring_parser.parse(docstring, docstring_parser.DocstringStyle.GOOGLE)

    # Function name
    short_description = parsed_docstring.short_description or ""
    output_dict[create_key(function_name=name, version=version)] = _parse_docstring_element_text(short_description)[0]

    # Function description
    if parsed_docstring.long_description:
        long_desc_key = create_key(function_name=name, description=True, version=version)
        output_dict[long_desc_key] = _convert_to_rendering_format(parsed_docstring.long_description)

    # Function parameters
    if parsed_docstring.params:
        _generate_key_for_parameters(name, output_dict, parsed_docstring.params, version)

    # Function return
    if parsed_docstring.returns:
        return_name = _parse_docstring_element_text(parsed_docstring.returns.description)[0]
        if return_name:
            return_key = create_key(function_name=name, output=True, version=version)
            output_dict[return_key] = return_name


# Add versioning if the function is versioned
def _generate_translation_mapping_for_functions(output_dict: dict, module):
    functions_to_export = getattr(module, COGNITE, [])
    functions_map = inspect.getmembers(module, inspect.isfunction)

    for name, function in filter(lambda f: f[0] in functions_to_export, functions_map):
        if versioning.is_versioned(function):
            _generate_key_for_versioned_function(function, output_dict)
        else:
            _generate_key_for_function(function, name, output_dict=output_dict)


# Process versioned function by adding only the deprecated version to the keys.
# The latest version does not have a version number in the key.
def _generate_key_for_versioned_function(function, output_dict):
    function_name = versioning.get_name(function)
    versions = versioning.get_versions(function_name)

    for version in versions:
        func = versioning.get(function_name, version)
        _generate_key_for_function(
            func,
            function_name,
            output_dict=output_dict,
            version=version if version != versions[-1] else None,
        )


# Create the output dictionary for the JSON file
def create_mapping_for_translations():
    """Create a the output dictionary for the JSON file."""
    output_dict = {}
    for _, module in inspect.getmembers(indsl, inspect.ismodule):
        toolbox_name = getattr(module, TOOLBOX_NAME, None)
        if toolbox_name is not None and toolbox_name != "Not listed operations":
            output_dict[create_key(toolbox=toolbox_name)] = toolbox_name

            _generate_translation_mapping_for_functions(output_dict, module)

    return output_dict


# Write the keys and values to the JSON file
def create_json_file():
    """Create mapping for InDSL and write to file."""
    output_dict = create_mapping_for_translations()
    file_path = "translated_operations.json"
    # limit the number of keys to 1000
    output_dict = {k: output_dict[k] for k in list(output_dict.keys())[:1000]}
    with open(file_path, "w") as file:
        json.dump(output_dict, file, indent=4)


# Compare keys_from_locize.json with the output of create_json_file() and make a new file with the differences
def compare_and_push_to_locize():
    """Push differences to locize."""
    # Pull data from locize
    pull_url = f"https://api.locize.app/export/{LOCIZE_PROJECT_ID}/latest/en/{NAMESPACE}"
    headers = {"Authorization": f"Bearer {LOCIZE_API_KEY}", "Content-Type": "application/json"}

    pull_response = requests.get(pull_url, headers=headers, timeout=30)
    if pull_response.status_code == 200:
        with open(OUTPUT_FILE, "w") as file:
            file.write(pull_response.text)
    else:
        pull_response.raise_for_status()

    # Get the keys from locize, this is the source of truth
    with open("keys_from_locize.json", "r") as file:
        keys_from_locize = json.load(file)

    # Get new keys and values with potential changes from the create_json_file() function
    with open("translated_operations.json", "r") as file:
        translated_operations = json.load(file)

    # Compare the keys and values from locize with the new keys and values and make a new file with the differences
    keys_diff = {}
    for key, value in translated_operations.items():
        if keys_from_locize.get(key) != value:
            keys_diff[key] = value

    # Push the keys_diff.json to locize
    if keys_diff != {}:
        push_response = requests.post(
            f"https://api.locize.app/update/{LOCIZE_PROJECT_ID}/latest/en/{NAMESPACE}",
            headers=headers,
            data=json.dumps(keys_diff),
            timeout=30,
        )
        push_response.raise_for_status()
    else:
        pass


if __name__ == "__main__":
    create_json_file()
    compare_and_push_to_locize()
