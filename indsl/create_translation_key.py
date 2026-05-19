PREFIX = "INDSL"
DESCRIPTION = "DESCRIPTION"
RETURN = "RETURN"
TOOLBOX = "TOOLBOX"


# Format the string to uppercase and replace spaces with underscores
def _format_str(s: str | None) -> str:
    return s.upper().replace(" ", "_") if s else ""


# Create keys for the JSON file
def create_key(
    toolbox: str | None = None,
    function_name: str | None = None,
    description: bool | None = False,
    parameter: str | None = None,
    output: bool | None = False,
    version: str | None = None,
) -> str | None:
    """Create keys for the JSON file."""
    elements = [PREFIX]

    if toolbox:
        elements.append(TOOLBOX)
        elements.append(_format_str(toolbox))
    else:
        if function_name:
            elements.append(_format_str(function_name))
            if output:
                elements.append(RETURN)
            elif parameter:
                elements.append(_format_str(parameter))
                if description:
                    elements.append(DESCRIPTION)
            elif description:
                elements.append(DESCRIPTION)

    version_str = f"_{_format_str(version.replace('.', '_'))}" if version else ""
    return "_".join(elements) + version_str if elements else None
