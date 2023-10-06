# Copyright 2023 Cognite AS
from .engineering_calcs import productivity_index
from .gas_density_calcs import calculate_gas_density
from .shut_in_detector import calculate_shutin_interval
from .shut_in_variables import calculate_shutin_variable
from .well_prod_status import calculate_well_prod_status


TOOLBOX_NAME = "Oil and gas"

__all__ = [
    "productivity_index",
    "calculate_shutin_interval",
    "calculate_shutin_variable",
    "calculate_well_prod_status",
    "calculate_gas_density",
]

__cognite__ = [
    "productivity_index",
    "calculate_shutin_interval",
    "calculate_shutin_variable",
    "calculate_well_prod_status",
    "calculate_gas_density",
]
