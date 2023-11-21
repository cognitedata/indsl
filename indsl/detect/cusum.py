# Copyright 2023 Cognite AS
from typing import Literal, Optional

import numpy as np
import pandas as pd

from indsl.exceptions import MATPLOTLIB_REQUIRED, UserValueError
from indsl.type_check import check_types
from indsl.validations import validate_series_has_time_index, validate_series_is_not_empty


class Cusum:
    """Cumulative sum (CUSUM).

    This technique calculates the cumulative sum of positive and negative changes (g+t and g−t) in the data (x) and compares them to a threshold.
    When this threshold is exceeded, a change is detected (time_talarm), and the cumulative sum restarts from zero.
    To avoid the detection of a change in absence of an actual change or a slow drift, this algorithm also depends on a parameter drift for drift correction.
    Remove extreme standalone outliers before using this technique to get better results.

    Attributes:
        data (pd.Series): Time series data.
        mean_data (pd.Series): Smoothed time series.
        threshold (float): Amplitid threshold.
        drift (float): Drift term.
        detect (str): Type of changes to detect.
        predict_ending (bool): Predict end point.
        alpha (float): Smoothing factor for mean data.
        time_alarm (np.array): Detected change points.
        time_alarm_initial (np.array): Start of the changes.
        time_alarm_final (np.array): End of the changes.
        cumulative_positive_sum (np.array): Cumulative sum of positive changes.
        cumulative_negative_sum (np.array): Cumulative sum of negative changes.

    References:
        https://nbviewer.org/github/demotu/detecta/blob/master/docs/detect_cusum.ipynb
    """

    @check_types
    def __init__(
        self,
        data: pd.Series,
        threshold: Optional[float] = None,
        drift: Optional[float] = None,
        detect: Literal["both", "increase", "decrease"] = "both",
        predict_ending: bool = True,
        alpha: float = 0.05,
    ):
        """Cusum init function.

        Args:
            data: Time series
            threshold: Amplitid threshold.
                Cumulative changes are compared to this threshold. Defaults to None.
                When this threshold is exceeded, a change is detected, and the cumulative sum restarts from 0.
                If the threshold is not provided, it is assigned to 5 * standard_deviation of the data.
            drift: Drift term.
                Prevents any change in the absence of change. Defaults to None.
                If fewer false alarms are wanted, try to increase `drift`.
                If the threshold is not provided, it is assigned to (2 * data_standard_deviation - data_mean) / 2.
            detect: Type of changes to detect:
                    * "both" for detecting both increasing and decreasing changes in the data (default)
                    * "increase" for detecting increasing changes in the data
                    * "decrease" for detecting decreasing changes in the data
            predict_ending: Predict end point.
                Prolongs the change until the predicted end point. Defaults to True.
                If false, single change points are detected.
            alpha: Smoothing factor.
                Value between 0 < alpha <= 1. Defaults to 0.05.

        Raises:
            UserTypeError: If a time series with the wrong index is provided.
            UserValueError: If an empty time series is passed into the function.
            UserValueError: If the alpha value is not in the range (0, 1].
        """
        # Check for correct alpha value
        if not (0 < alpha <= 1):
            raise UserValueError(f"Invalid alpha value: {alpha}. Alpha should be in the range (0, 1].")

        self.data = data
        self._validate_time_series()

        # user inputs
        self.threshold = 5 * data.std() if not threshold else threshold
        self.drift = (2 * data.std() - data.mean()) / 2 if not drift else drift
        self.detect = detect
        self.predict_ending = predict_ending
        self.alpha = alpha

        # smooth time series used as mean data
        self.mean_data = data.ewm(alpha=alpha).mean()

        # initialize empty results
        self.time_alarm = np.array([])
        self.time_alarm_initial = np.array([])
        self.time_alarm_final = np.array([])
        self.cumulative_positive_sum = np.array([])
        self.cumulative_negative_sum = np.array([])

    def _validate_time_series(self):
        """Validate time series.

        Raises:
            UserValueError: If an empty time series is passed into the function.
            UserValueError: If a time series with the wrong index is provided.
        """
        validate_series_is_not_empty(self.data)
        validate_series_has_time_index(self.data)

        if not self.data.index.is_monotonic_increasing:
            raise UserValueError("Time series index is not increasing.")

    @check_types
    def _detect_changes(self, endings: bool = False):
        """Detection of change points.

        Args:
            endings: Detect only the end of the changes previously detected.
                Defaults to False.
        """
        if endings:  # revert the datapoints
            x = self.data.values[::-1]
            m = self.mean_data.values[::-1]

        else:
            x = self.data.values
            m = self.mean_data.values

        cumulative_positive, cumulative_negative = np.zeros(x.size), np.zeros(x.size)
        time_alarm, time_alarm_initial, time_alarm_final = (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
        time_alarm_positive, time_alarm_negative = 0, 0
        s = x - m

        for i in range(1, x.size):
            cumulative_positive[i] = cumulative_positive[i - 1] + s[i] - self.drift  # cumulative sum for + change
            cumulative_negative[i] = cumulative_negative[i - 1] - s[i] - self.drift  # cumulative sum for - change

            if cumulative_positive[i] < 0:
                cumulative_positive[i], time_alarm_positive = 0, i
            if cumulative_negative[i] < 0:
                cumulative_negative[i], time_alarm_negative = 0, i

            # detect changes
            if self.detect == "increase" and (cumulative_positive[i] > self.threshold):
                time_alarm = np.append(time_alarm, i)  # alarm index for a positive change
                time_alarm_initial = np.append(time_alarm_initial, time_alarm_positive)  # start
                cumulative_positive[i], cumulative_negative[i] = 0, 0  # reset alarm
            elif self.detect == "decrease" and (cumulative_negative[i] > self.threshold):
                time_alarm = np.append(time_alarm, i)  # alarm index for a negative change
                time_alarm_initial = np.append(time_alarm_initial, time_alarm_negative)  # start
                cumulative_positive[i], cumulative_negative[i] = 0, 0  # reset alarm
            elif self.detect == "both" and (
                cumulative_positive[i] > self.threshold or cumulative_negative[i] > self.threshold
            ):
                time_alarm = np.append(time_alarm, i)  # alarm index for positive or negative change
                time_alarm_initial = np.append(
                    time_alarm_initial,
                    time_alarm_positive if cumulative_positive[i] > self.threshold else time_alarm_negative,
                )  # start
                cumulative_positive[i], cumulative_negative[i] = 0, 0  # reset alarm

        # Eliminate repeated changes, changes that have the same beginning
        time_alarm_initial, ind = np.unique(time_alarm_initial, return_index=True)
        time_alarm = time_alarm[ind]

        # store results as attributes
        if not endings:
            self.time_alarm = time_alarm
            self.time_alarm_initial = time_alarm_initial
            self.time_alarm_final = time_alarm_final

            self.cumulative_positive_sum = pd.Series(cumulative_positive, index=self.data.index)
            self.cumulative_negative_sum = pd.Series(cumulative_negative, index=self.data.index)

        else:
            self.time_alarm_final = self.data.values.size - time_alarm_initial[::-1] - 1

    def _detect_ending(self):
        """Estimation of when the change ends (offline form)."""
        self._detect_changes(endings=True)

        self._fix_time_alarm_size_errors()

    def _fix_time_alarm_size_errors(self):
        if self.time_alarm_initial.size != self.time_alarm_final.size:
            if self.time_alarm_initial.size < self.time_alarm_final.size:
                self.time_alarm_final = self.time_alarm_final[
                    [np.argmax(self.time_alarm_final >= i) for i in self.time_alarm]
                ]
            else:
                ind: np.array = [np.argmax(i >= self.time_alarm[::-1]) - 1 for i in self.time_alarm_final]
                self.time_alarm = self.time_alarm[ind]
                self.time_alarm_initial = self.time_alarm_initial[ind]
        # Delete intercalated changes (the ending of the change is after
        # the beginning of the next change)
        ind = self.time_alarm_final[:-1] - self.time_alarm_initial[1:] > 0
        if ind.any():
            self.time_alarm = self.time_alarm[~np.append(False, ind)]
            self.time_alarm_initial = self.time_alarm_initial[~np.append(False, ind)]
            self.time_alarm_final = self.time_alarm_final[~np.append(ind, False)]

    def _plot_cusum(self) -> None:
        """Plot cusum results.

        Plots 2 figures containing the results of the cusum function:
        1. Raw data, smoothed data, detected change points, start of the change and end of the change (if predict_ending = True).
        2. Cumulative sums for increasing and decreasing changes.
        """
        try:
            from matplotlib import pyplot as plt
        except ImportError:
            raise ImportError(MATPLOTLIB_REQUIRED)
        else:
            _, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            time_alarm_initial_index = [pd.Timestamp(self.data.index[tai_val]) for tai_val in self.time_alarm_initial]
            time_alarm_final_index = [pd.Timestamp(self.data.index[taf_val]) for taf_val in self.time_alarm_final]
            time_alarm_index = [pd.Timestamp(self.data.index[ta_val]) for ta_val in self.time_alarm]

            ax1.plot(self.data.index, self.data.values, "-go", lw=2, label="Raw data")
            ax1.plot(self.data.index, self.mean_data.values, "-o", color="orange", lw=2, label="Smoothed data")

            if len(self.time_alarm):
                ax1.plot(
                    time_alarm_initial_index,
                    self.data.iloc[self.time_alarm_initial],
                    ">",
                    mfc="b",
                    mec="g",
                    ms=10,
                    label="Start",
                )
                if self.predict_ending:
                    ax1.plot(
                        time_alarm_final_index,
                        self.data.values[self.time_alarm_final],
                        "<",
                        mfc="b",
                        mec="g",
                        ms=10,
                        label="Ending",
                    )
                ax1.plot(
                    time_alarm_index,
                    self.data.values[self.time_alarm],
                    "o",
                    mfc="r",
                    mec="m",
                    mew=1,
                    ms=10,
                    label="Alarm",
                )
                ax1.legend(loc="best", framealpha=0.5, numpoints=1)

            ymin, ymax = (
                self.data.values[np.isfinite(self.data.values)].min(),
                self.data.values[np.isfinite(self.data.values)].max(),
            )
            yrange = ymax - ymin if ymax > ymin else 1
            ax1.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
            ax1.set_title(
                f"Detected changes in the data (threshold= {self.threshold}, drift= {self.drift}): N changes ={len(self.time_alarm_initial)}"
            )
            ax2.plot(self.data.index, self.cumulative_positive_sum, "-oy", label="Increasing changes", lw=0.8)
            ax2.plot(self.data.index, self.cumulative_negative_sum, "-om", label="Decreasing changes", lw=0.8)

            # ax2.set_ylim(-0.01*threshold, 1.1*threshold)
            ax2.axhline(self.threshold, color="r", label=f"Threshold at {self.threshold}")
            ax2.set_title("Cumulative sums of increasing and decreasing changes.")
            ax2.legend(loc="best", framealpha=0.5, numpoints=1)
            plt.tight_layout()
            plt.show()

    def _create_binary_output(self) -> pd.Series:
        """Binary pd.Series with the detected changes.

        Returns:
            pd.Series: Binary time series.
            Change detected = 1, No change detected = 0.
        """
        output_series: pd.Series = pd.Series([0] * self.data.size, index=self.data.index)

        if len(self.time_alarm_final) > 0:
            # amp = data[taf] - data[tai]
            for i in range(len(self.time_alarm_initial)):
                output_series[self.time_alarm_initial[i] : self.time_alarm_final[i] + 1] = 1
        else:
            output_series[self.time_alarm_initial] = 1

        return output_series

    @check_types
    def cusum(
        self,
        plot_cusum: bool = False,
    ) -> pd.Series:
        """Cumulative sum (CUSUM).

        This technique calculates the cumulative sum of positive and negative changes (g+t and g−t) in the data (x) and compares them to a threshold.
        When this threshold is exceeded a change is detected (ttalarm) and the cumulative sum restarts from zero.
        To avoid the detection of a change in absence of an actual change or a slow drift, this algorithm also depends on a parameter drift for drift correction.
        To get better results, it is recommended to remove extreme standalone outliers before using this technique.

        Returns:
            pd.Series: Binary time series.
                Change detected = 1, No change detected = 0.

        Raises:
            UserTypeError: If a time series with the wrong index is provided.
            UserValueError: If an empty time series is passed into the function.

        References:
            https://nbviewer.org/github/demotu/detecta/blob/master/docs/detect_cusum.ipynb
        """
        self._detect_changes()

        if self.time_alarm_initial.size and self.predict_ending:
            self._detect_ending()
        if plot_cusum:
            self._plot_cusum()
        return self._create_binary_output()


@check_types
def cusum(
    data: pd.Series,
    threshold: Optional[float] = None,
    drift: Optional[float] = None,
    detect: Literal["both", "increase", "decrease"] = "both",
    predict_ending: bool = True,
    alpha: float = 0.05,
    return_series_type: Literal[
        "cusum_binary_result", "mean_data", "positive_cumulative_sum", "negative_cumulative_sum"
    ] = "cusum_binary_result",
) -> pd.Series:
    """Cumulative sum (CUSUM).

    This technique calculates the cumulative sum of positive and negative changes (g+t and g−t) in the data (x) and compares them to a threshold.
    When this threshold is exceeded, a change is detected (ttalarm), and the cumulative sum restarts from zero.
    To avoid the detection of a change in absence of an actual change or a slow drift, this algorithm also depends on a parameter `drift` for drift correction.
    Remove extreme standalone outliers before using this technique to get a better result.

    Typical uses of this function:

    1. Set the type of series to return to "mean_data" to visualize the smoothed data. Leave the rest of the parameters to their default values.
    2. Adjust the alpha parameter to get the desired smoothing for the data.
    3. Set the type of series to return to "positive_cumulative_sum" or "negative_cumulative_sum" to visualize the cumulative sum of the positive or negative changes.
    4. Adjust the threshold and drift accordingly to get the desired number of change points.
    5. Set the type of series to return to "cusum_binary_result" to visualize the detected changes.

    Args:
        data: Time series.
        threshold: Amplitude threshold.
            Cumulative changes are compared to this threshold. Defaults to None.
            When this is exceeded a change is detected and the cumulative sum restarts from 0.
            If the threshold is not provided, it is assigned to 5 * standard_deviation of the data.
        drift: Drift term.
            Prevents any change in the absence of change. Defaults to None.
            If fewer false alarms are wanted, try to increase `drift`.
            If the threshold is not provided, it is assigned to (2 * data_standard_deviation - data_mean) / 2.
        detect: Type of changes to detect.
                Options are:
                * "both" for detecting both increasing and decreasing changes in the data (default)
                * "increase" for detecting increasing changes in the data
                * "decrease" for detecting decreasing changes in the data
        predict_ending: Predict end point.
            Prolongs the change until the predicted end point. Defaults to True.
            If false, single change points are detected.
        alpha: Smoothing factor.
            Value between 0 < alpha <= 1. Defaults to 0.05.
        return_series_type: Type of series to return.
            Defaults to "cusum_binary_result".
            This option allows the user to visualize the intermediate steps of the algorithm.
            Options are:

            * "cusum_binary_result" returns the cusum results as a binary time series. Change detected = 1, No change detected = 0.
            * "mean_data" returns the smoothed data.
            * "positive_cumulative_sum" returns the positive cumulative sum.
            * "negative_cumulative_sum" returns the negative cumulative sum.

    Returns:
        pd.Series: Time series.
            Specified in the return_series_type parameter.

    Raises:
        UserTypeError: If a time series with the wrong index is provided.
        UserValueError: If an empty time series is passed into the function.

    References:
        https://nbviewer.org/github/demotu/detecta/blob/master/docs/detect_cusum.ipynb
    """
    cusum_detection = Cusum(
        data=data,
        threshold=threshold,
        drift=drift,
        detect=detect,
        predict_ending=predict_ending,
        alpha=alpha,
    )

    if return_series_type == "mean_data":
        return cusum_detection.mean_data
    else:
        cusum_binary_result = cusum_detection.cusum(plot_cusum=False)

        if return_series_type == "positive_cumulative_sum":
            return cusum_detection.cumulative_positive_sum
        elif return_series_type == "negative_cumulative_sum":
            return cusum_detection.cumulative_negative_sum
        return cusum_binary_result
