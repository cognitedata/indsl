# Copyright 2023 Cognite AS
from .flag_detection import circulation_detection, inhole_detection, onbottom_detection, rotation_detection


TOOLBOX_NAME = "drilling"

__all__ = ["rotation_detection", "onbottom_detection", "inhole_detection", "circulation_detection"]


__cognite__ = ["rotation_detection", "onbottom_detection", "inhole_detection", "circulation_detection"]
