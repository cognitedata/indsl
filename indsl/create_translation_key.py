from typing import Optional


PREFIX = "INDSL"
DESCRIPTION = "DESCRIPTION"
RETURN = "RETURN"
TOOLBOX = "TOOLBOX"


# Format the string to uppercase and replace spaces with underscores
def _format_str(s: Optional[str]) -> str:
    return s.upper().replace(" ", "_") if s else ""


# Create keys for the JSON file
def create_key(
    toolbox: Optional[str] = None,
    function_name: Optional[str] = None,
    description: Optional[bool] = False,
    parameter: Optional[str] = None,
    output: Optional[bool] = False,
    version: Optional[str] = None,
) -> Optional[str]:
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

    version_str = f"_{_format_str(version)}" if version else ""
    return "_".join(elements) + version_str if elements else None
