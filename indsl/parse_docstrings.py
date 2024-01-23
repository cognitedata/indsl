import inspect
import docstring_parser
import indsl
import json

PREFIX = "INDSL"


def docstring_to_json(module):
    output_dict = {}
    for _, module in inspect.getmembers(indsl):
        toolbox_name = getattr(module, "TOOLBOX_NAME", None)
        if toolbox_name is not None:
            output_dict[PREFIX + "_" + toolbox_name.upper().replace(" ", "_")] = toolbox_name

        # extract doctring from each function
        functions_to_export = getattr(module, "__cognite__", [])
        functions_map = inspect.getmembers(module, inspect.isfunction)
        for name, function in functions_map:
            if name in functions_to_export:
                docstring = str(function.__doc__) if function.__doc__ else ""
                parsed_docstring = docstring_parser.parse(docstring, docstring_parser.DocstringStyle.GOOGLE)
                short_description = parsed_docstring.short_description if parsed_docstring.short_description else ""
                # remove "." from the end of the names if they're in short_description, but keep them for the descriptions
                output_dict[PREFIX + "_" + name.upper().replace(" ", "_")] = short_description

    with open("toolboxes.json", "w") as f:
        json.dump(output_dict, f, indent=4)
