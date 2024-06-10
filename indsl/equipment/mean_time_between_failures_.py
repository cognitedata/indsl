import pandas as pd

from indsl.type_check import check_types


@check_types
def mean_time_between_failures(mean_time_to_failure: pd.Series, mean_time_to_resolution: pd.Series) -> pd.Series:
    r"""Mean time between failures.

    Calculate the mean time between failures (MTBF) of a system based on
    the sum of the mean time to failure (MTTF) and mean time to resolution
    (MTTRes). The MTBF is given by

    .. math::
      MTBF = MTTF + MTTRes = \frac{1}{\lambda} + \frac{1}{\mu}

    with MTTF and MTTRes written as

    .. math::
      MTTF = \frac{1}{\lambda}

    and

    .. math::
      MTTRes = \frac{1}{\mu}

    where :math:`\lambda` is the failure rate and :math:`\mu` is the repair rate.

    Args:
      mean_time_to_failure: Mean time to failure.
          Time series data of the mean time to failure.
      mean_time_to_resolution: Mean time to resolution.
          Time series data of the mean time to resolution.

    Returns:
      pd.Series
        Time series data of the MTBF of the system.
    """
    mtbf = mean_time_to_failure + mean_time_to_resolution

    return mtbf
