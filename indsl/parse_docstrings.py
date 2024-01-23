import inspect
import docstring_parser
import indsl


PREFIX = "INDSL"


def docstring_to_json(module):
    with open("toolboxes.json", "w") as f:
        for _, module in inspect.getmembers(indsl):
            toolbox_name = getattr(module, "TOOLBOX_NAME", None)
            if toolbox_name is not None:
                f.write(PREFIX + "_" + toolbox_name.upper().replace(" ", "_") + "=" + '"' + toolbox_name + '"' + "\n")

            # extract doctring from each function
            functions_to_export = getattr(module, "__cognite__", [])
            functions_map = inspect.getmembers(module, inspect.isfunction)
            for name, function in functions_map:
                if name in functions_to_export:
                    docstring = str(function.__doc__) if function.__doc__ else ""
                    parsed_docstring = docstring_parser.parse(docstring, docstring_parser.DocstringStyle.GOOGLE)
                    f.write(
                        PREFIX
                        + "_"
                        + name.upper().replace(" ", "_")
                        + "="
                        + '"'
                        + str(parsed_docstring.short_description)
                        + '"'
                        + "\n"
                    )
