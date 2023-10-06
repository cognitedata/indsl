from __future__ import annotations

import importlib

from inspect import getmembers, isfunction, ismodule

from models import Operation, OperationVersion


class DataScienceService:
    def __init__(self):
        import sys

        print(sys.path)
        self.module = importlib.import_module("indsl")
        self._operations = self._get_all_operations()

    def get_available_operations(self):
        """Return list of all operations"""
        return self._operations.values()

    def get_operation_by_op_code(self, op_code: str, version: str) -> OperationVersion:
        return self._operations[op_code].get_version(version)

    def _get_all_operations(self):
        """Return a mapping from op_code -> Operation"""
        operations = {}
        for toolbox, toolbox_name in self._get_toolboxes_from_package():
            for function in self._get_functions_from_toolbox(toolbox):
                operation = Operation(base_function=function, category=toolbox_name)
                if operation.op in operations:
                    raise RuntimeError(f"Detected two operations with the same op code '{operation.op}'")
                operations[operation.op] = operation
        return operations

    def _get_functions_from_toolbox(self, toolbox: type.ModuleType):  # type: ignore
        functions_map = getmembers(toolbox, isfunction)
        functions_to_export = getattr(toolbox, "__cognite__", [])
        for name, function in functions_map:
            if name in functions_to_export:
                yield function

    def _get_toolboxes_from_package(self):
        module_tuples = getmembers(self.module, ismodule)
        for name, module in module_tuples:
            toolbox_name = getattr(module, "TOOLBOX_NAME", None)
            if toolbox_name is not None:
                yield module, toolbox_name


data_science_service = DataScienceService()
