import inspect
import docstring_parser
import indsl
import json

PREFIX = "INDSL"
PARAMETER = "PARAMETER"
DESCRIPTION = "DESCRIPTION"
RETURN = "RETURN"
TOOLBOX_NAME = "TOOLBOX_NAME"
COGNITE = "__cognite__"


def create_key(*args):
    return "_".join(arg.upper().replace(" ", "_") for arg in args)


def create_value(value):
    return value.replace("\n", " ").replace(".", "").replace("_", " ").capitalize()


def docstring_to_json(module):
    output_dict = {}
    for _, module in inspect.getmembers(indsl):
        toolbox_name = getattr(module, TOOLBOX_NAME, None)
        if toolbox_name is not None:
            output_dict[create_key(PREFIX, toolbox_name)] = toolbox_name

        # extract doctring from each function
        functions_to_export = getattr(module, COGNITE, [])
        functions_map = inspect.getmembers(module, inspect.isfunction)
        for name, function in functions_map:
            if name in functions_to_export:
                docstring = str(function.__doc__) if function.__doc__ else ""
                parsed_docstring = docstring_parser.parse(docstring, docstring_parser.DocstringStyle.GOOGLE)
                short_description = parsed_docstring.short_description if parsed_docstring.short_description else ""
                output_dict[create_key(PREFIX, name)] = create_value(short_description)

                # Parameter names and descriptions
                parameters = parsed_docstring.params if parsed_docstring.params else []
                for parameter in parameters:
                    description = parameter.description if parameter.description else ""
                    output_dict[create_key(PREFIX, parameter.arg_name, PARAMETER)] = create_value(parameter.arg_name)
                    output_dict[create_key(PREFIX, parameter.arg_name, PARAMETER, DESCRIPTION)] = description.replace(
                        "\n", " "
                    ).replace(parameter.arg_name, "")

                # Name of the return value
                if parsed_docstring.returns:
                    return_names = parsed_docstring.returns.return_name
                    if return_names:
                        for return_name in return_names:
                            if return_name:
                                output_dict[create_key(PREFIX, return_name, RETURN)] = create_value(return_name)

    with open("toolboxes.json", "w") as f:
        json.dump(output_dict, f, indent=4)
