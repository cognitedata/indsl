# Copyright 2023 Cognite AS
from typing import Optional

import numpy as np
import pandas as pd

from indsl.exceptions import UserValueError
from indsl.type_check import check_types


class TimeSampler:
    """TimeSampler determines how and when samples will be taken from signal and noise.

    Samples timestamps for regular and irregular time signals
    Parameters:
    ----------
    start_time: float/int (default 0)
                Time sampling of time series starts
    stop_time: float/int (default 10)
                Time sampling of time series stops
    """

    def __init__(self, start_time=0, stop_time=10):
        """Initialize TimeSampler."""
        self.start_time = start_time
        self.stop_time = stop_time

    def sample_regular_time(self, num_points=None, resolution=None):
        """Sample regular time.

        Samples regularly spaced time using the number of points or the
        resolution of the signal. Only one of the parameters is to be
        initialized. The resolution keyword argument is given priority.

        Parameters:
        ----------
        num_points: int (default None)
            Number of points in time series
        resolution: float/int (default None)
            Resolution of the time series

        Returns:
        -------
        numpy array
            Regularly sampled timestamps
        """
        if num_points is None and resolution is None:
            raise ValueError("One of the keyword arguments must be initialized.")
        if resolution is not None:
            time_vector = np.arange(self.start_time, self.stop_time, resolution)
            return time_vector
        else:
            time_vector = np.linspace(self.start_time, self.stop_time, num_points)
            return time_vector

    def sample_irregular_time(self, num_points=None, resolution=None, keep_percentage=100):
        """Sample irregular time.

        Samples irregularly spaced time using the number of points or the
        resolution of the signal. Only one of the parameters is to be
        initialized. The resolution keyword argument is given priority.

        Parameters:
        ----------
        num_points: int (default None)
            Number of points in time series
        resolution: float/int (default None)
            Resolution of the time series
        keep_percentage: int(default 100)
            Percentage of points to be retained in the irregular series

        Returns:
        -------
        numpy array
            Irregularly sampled timestamps
        """
        if num_points is None and resolution is None:
            raise ValueError("One of the keyword arguments must be initialized.")
        if resolution is not None:
            time_vector = np.arange(self.start_time, self.stop_time, resolution)
        else:
            time_vector = np.linspace(self.start_time, self.stop_time, num_points)
            resolution = float(self.stop_time - self.start_time) / num_points
        time_vector = self._select_random_indices(time_vector, keep_percentage)
        return self._create_perturbations(time_vector, resolution)

    def _create_perturbations(self, time_vector, resolution):
        """Internal functions to create perturbations in timestamps.

        Parameters:
        ----------
        time_vector: numpy array
            timestamp vector
        resolution: float/int
            resolution of the time series

        Returns:
        -------
        numpy array
            Irregularly sampled timestamps with perturbations
        """
        sample_perturbations = np.random.normal(loc=0.0, scale=resolution, size=len(time_vector))
        time_vector = time_vector + sample_perturbations
        return np.sort(time_vector)

    def _select_random_indices(self, time_vector, keep_percentage):
        """Internal functions to randomly select timestamps.

        Parameters:
        ----------
        time_vector: numpy array
            timestamp vector
        keep_percentage: float/int
            percentage of points retained

        Returns:
        -------
        numpy array
            Irregularly sampled timestamps
        """
        num_points = len(time_vector)
        num_select_points = int(keep_percentage * num_points / 100)
        index = np.sort(np.random.choice(num_points, size=num_select_points, replace=False))
        return time_vector[index]


class Sinusoidal:
    """Signal generator for harmonic (sinusoidal) waves.

    Parameters:
    ----------
    amplitude : number (default 1.0)
        Amplitude of the harmonic series
    frequency : number (default 1.0)
        Frequency of the harmonic series
    ftype : function (default np.sin)
        Harmonic function
    """

    def __init__(self, amplitude=1.0, frequency=1.0, ftype=np.sin):
        """Initialize the signal generator."""
        self.vectorizable = True
        self.amplitude = amplitude
        self.ftype = ftype
        self.frequency = frequency

    def sample_next(self, time, samples, errors):
        """Sample a single time point.

        Parameters:
        ----------
        time : number
            Time at which a sample was required
        samples : array-like
            all samples taken so far
        errors : array-like
            all errors taken so far

        Returns:
        -------
        float
            sampled signal for time t
        """
        return self.amplitude * self.ftype(2 * np.pi * self.frequency * time)

    def sample_vectorized(self, time_vector):
        """Sample entire series based off of time vector.

        Parameters:
        ----------
        time_vector : array-like
            Timestamps for signal generation

        Returns:
        -------
        array-like
            sampled signal for time vector
        """
        if self.vectorizable is True:
            signal = self.amplitude * self.ftype(2 * np.pi * self.frequency * time_vector)
            return signal
        else:
            raise ValueError("Signal type not vectorizable")


class TimeSeries:
    """A TimeSeries object is the main interface from which to sample time series.

    You have to provide at least a signal generator; a noise generator is optional.
    It is recommended to set the sampling frequency.

    Parameters:
    ----------
    signal_generator : Signal object
        signal object for time series
    noise_generator : Noise object
        noise object for time series
    """

    def __init__(self, signal_generator, noise_generator=None):
        """Initialize the TimeSeries object."""
        self.signal_generator = signal_generator
        self.noise_generator = noise_generator

    def sample(self, time_vector):
        """Samples from the specified TimeSeries.

        Parameters:
        ----------
        time_vector : numpy array
            Times at which to generate a sample

        Returns:
        -------
        samples, signals, errors, : tuple (array, array, array)
            Returns samples, and the signals and errors they were constructed from
        """
        # Vectorize if possible
        if (
            self.signal_generator.vectorizable
            and self.noise_generator is not None
            and self.noise_generator.vectorizable
        ):
            signals = self.signal_generator.sample_vectorized(time_vector)
            errors = self.noise_generator.sample_vectorized(time_vector)
            samples = signals + errors
        elif self.signal_generator.vectorizable and self.noise_generator is None:
            signals = self.signal_generator.sample_vectorized(time_vector)
            errors = np.zeros(len(time_vector))
            samples = signals
        else:
            n_samples = len(time_vector)
            samples = np.zeros(n_samples)  # Signal and errors combined
            signals = np.zeros(n_samples)  # Signal samples
            errors = np.zeros(n_samples)  # Handle errors seprately

            # Sample iteratively, while providing access to all previously sampled steps
            for i in range(n_samples):
                # Get time
                t = time_vector[i]
                # Sample error
                if self.noise_generator is not None:
                    errors[i] = self.noise_generator.sample_next(t, samples[: i - 1], errors[: i - 1])

                # Sample signal
                signal = self.signal_generator.sample_next(t, samples[: i - 1], errors[: i - 1])
                signals[i] = signal

                # Compound signal and noise
                samples[i] = signals[i] + errors[i]

        # Return both times and samples, as well as signals and errors
        return samples, signals, errors


class BaseNoise:
    """BaseNoise class.

    Signature for all noise classes.
    """

    def __init__(self):
        """Initialize the BaseNoise object."""
        raise NotImplementedError

    def sample_next(self, t, samples, errors):  # We provide t for irregularly sampled timeseries
        """Samples next point based on history of samples and errors.

        Parameters:
        ----------
        t : int
            time
        samples : array-like
            all samples taken so far
        errors : array-like
            all errors sampled so far

        Returns:
        -------
        float
            sampled error for time t
        """
        raise NotImplementedError


class RedNoise(BaseNoise):
    """Red noise generator.

    This class adds correlated (red) noise to your signal.

    Attributes:
    ----------
    mean : float
        mean for the noise
    std : float
        standard deviation for the noise
    tau : float
        ?
    start_value : float
        ?
    """

    def __init__(self, mean=0, std=1.0, tau=0.2, start_value=0):
        """Initialize the RedNoise object."""
        self.vectorizable = False
        self.mean = mean
        self.std = std
        self.start_value = 0
        self.tau = tau
        self.previous_value = None
        self.previous_time = None

    def sample_next(self, t, samples, errors):
        """Samples next point based on history of samples and errors.

        Parameters:
        ----------
        t : int
            time
        samples : array-like
            all samples taken so far
        errors : array-like
            all errors sampled so far

        Returns:
        -------
        float
            sampled error for time t
        """
        if self.previous_time is None:
            red_noise = self.start_value
        else:
            time_diff = t - self.previous_time
            wnoise = np.random.normal(loc=self.mean, scale=self.std, size=1)
            red_noise = (self.tau / (self.tau + time_diff)) * (time_diff * wnoise + self.previous_value)
        self.previous_time = t
        self.previous_value = red_noise
        return red_noise


@check_types
def white_noise(
    data: pd.Series,
    snr_db: float = 30,
    seed: Optional[int] = None,
) -> pd.Series:
    """Add white noise.

    Adds white noise to the original data using a given signal-to-noise ratio (SNR).

    Args:
        data: Time series
        snr_db: SNR.
            Signal-to-noise ratio (SNR) in decibels. SNR is a comparison of the level of a signal to the level of
            background noise. SNR is defined as the ratio of signal power to noise power. A ratio higher than 1
            indicates more signal than noise. Defaults to 30.
        seed: Seed.
            A seed (integer number) to initialize the random number generator. If left empty, then a fresh,
            unpredictable value will be generated. If a value is entered, the exact random noise will be generated if
            the time series data and date range are not changed.

    Returns:
        pandas.Series: Output
            Original data plus white noise.
    """
    data_power = data.var()
    # Linear SNR
    try:
        snr = 10.0 ** (snr_db / 10.0)
    except OverflowError:
        raise UserValueError(f"snr_db value of {snr_db} is too large and causes overflow.")
    noise_power = data_power / snr
    rng = np.random.default_rng(seed)
    white_noise = np.sqrt(noise_power) * rng.standard_normal(len(data))

    return data + white_noise
