import enum
import typing

import numpy as np
import pandas as pd

from marshmallow import Schema, fields, post_dump
from typing_utils import to_json_type


class Type(fields.Field):
    """
    Translation between "frontend types" and Python types
    """

    def _serialize(self, value: typing.Any, attr: str, obj: typing.Any, **kwargs):
        try:
            return to_json_type(value)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Attempt to serialize unsupported type: {value}. {e}")


class InputOutputType(fields.Field):
    """
    Representation of input type ("ts", "result" or "const")
    """

    def _serialize(self, value: typing.Any, attr: str, obj: typing.Any, **kwargs):
        # XXX No way to determine it from function annotation or docstring. Semi-hardcoded behaviour here.
        try:
            if issubclass(value, pd.Series):
                return ["ts"]
        except TypeError:
            pass
        return ["ts", "const"]


class EnumInstance(Schema):
    label = fields.String(attribute="name")
    value = fields.String()


class OperationParameterSchema(Schema):
    name = fields.String()
    type = Type()
    default_value = fields.Method("get_default")
    description = fields.String()
    param = fields.String()
    options = fields.Method("get_options")
    expected_pattern = fields.String()

    def get_default(self, parameter):
        d = getattr(parameter, "default_value", None)
        if isinstance(d, enum.Enum):
            return d.value
        elif isinstance(d, (pd.Timestamp, pd.Timedelta)):
            return str(d)
        elif isinstance(d, float) and np.isinf(d):
            return np.sign(d) * (1e14 - 1)
        return d

    def get_options(self, parameter):
        if typing.get_origin(parameter.type) == typing.Literal:
            valid_arguments = typing.get_args(parameter.type)
            return [{"label": v, "value": v} for v in valid_arguments]

        elif typing.get_origin(parameter.type) is None and issubclass(parameter.type, enum.Enum):
            return EnumInstance().dump(parameter.type, many=True)


class OperationInputSchema(OperationParameterSchema):
    types = InputOutputType(attribute="type")

    class Meta:
        fields = ("name", "description", "types", "param")
        ordered = True


class OperationOutputSchema(OperationParameterSchema):
    types = InputOutputType(attribute="type")

    class Meta:
        fields = ("name", "description", "types")
        ordered = True


class OperationVersionSchema(Schema):
    name = fields.String(required=True)
    version = fields.String(required=True)
    description = fields.String()
    deprecated = fields.Boolean(required=True)
    changelog = fields.String()
    inputs = fields.List(fields.Nested(OperationInputSchema()))
    outputs = fields.List(fields.Nested(OperationOutputSchema()))
    parameters = fields.List(fields.Nested(OperationParameterSchema()))

    class Meta:
        ordered = True


class OperationSchema(Schema):
    op = fields.String(required=True)
    category = fields.String(required=True)
    versions = fields.List(fields.Nested(OperationVersionSchema()), required=True)

    class Meta:
        ordered = True

    @post_dump(pass_many=True)
    def postProcess(self, data, many):
        include_deprecated = self.context.get("include_deprecated", False)

        if include_deprecated:
            return data

        if not many:
            data = [data]

        for op in data:
            for version in op["versions"]:
                if version["deprecated"]:
                    op["versions"].remove(version)
            if len(op["versions"]) == 0:
                data.remove(op)

        if not many:
            data = data[0] if len(data) > 0 else None

        return data
