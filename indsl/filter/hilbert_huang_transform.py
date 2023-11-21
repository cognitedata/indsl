from typing import Optional

import numpy as np
import pandas as pd

from scipy.signal import hilbert

from indsl.ts_utils import get_timestamps


# DROP_SENTINAL is used to get the smallest (most negative)
# number that can be represented with a 32-bit signed integer.
DROP_SENTINAL = np.iinfo(np.int32).min


def _compute_hilbert_huang_2d_spectrum(values_X, values_Z, x_boundaries):
    """Compute 2D Hilbert-Huang spectrum.

    Calculate a 2D Hilbert-Huang power distribution in sparse format.
    This utility function generates a sparse matrix that captures a two-dimensional
    power distribution.
    """
    from sparse import COO

    # Calculates bin indices for values_X using the np.digitize() function.
    # This function returns an array of indices that can be used to bin the
    # input values_X into the bins defined by x_boundaries. Any values in
    # values_X that are out of the range defined by x_boundaries are assigned
    # the DROP_SENTINAL value.
    primary_indices = np.digitize(values_X, x_boundaries) - 1
    out_of_bounds = np.logical_or(values_X < x_boundaries[0], values_X >= x_boundaries[-1])
    primary_indices[out_of_bounds] = DROP_SENTINAL

    # Create meshgrids for time and IMF indices
    time_indices, imf_indices = np.meshgrid(np.arange(values_X.shape[0]), np.arange(values_X.shape[1]), indexing="ij")

    # Flatten arrays for COO matrix creation
    flattened_coordinates = np.vstack([primary_indices.ravel(), time_indices.ravel(), imf_indices.ravel()])
    exclusions = np.any(flattened_coordinates == DROP_SENTINAL, axis=0)

    # Apply the exclusions mask to values_Z
    values_Z_masked = np.ma.masked_array(values_Z, mask=exclusions.reshape(values_Z.shape)).compressed()

    # Adjust coordinates for valid points
    valid_coordinates = flattened_coordinates[:, ~exclusions]

    # Create the sparse COO matrix
    spectrum_shape = (x_boundaries.shape[0] - 1, values_X.shape[0], values_X.shape[1])
    sparse_spectrum = COO(valid_coordinates, values_Z_masked, shape=spectrum_shape)

    return sparse_spectrum


def _calculate_histogram_bins_and_centers(dataset, margin=1e-8):
    """Calculate histogram bins and centers.

    Determine bin edges for a histogram based on the provided dataset.
    """
    # Determine the data range with margin
    data_lower_bound = dataset.min() - margin
    data_upper_bound = dataset.max() + margin

    # Calculate the number of bins
    bin_count = min(int(np.sqrt(dataset.size)), 2048)  # Assuming a bin limit of 2048

    # Establish bin edges
    bin_edges = np.linspace(data_lower_bound, data_upper_bound, bin_count + 1)

    return bin_edges


def _hilbert_huang_spectrum(
    instant_freq,
    instant_amp,
    use_sparse_format=True,
):
    """Compute a Hilbert-Huang Spectrum (HHT).

    Calculate the Hilbert-Huang Transform (HHT) from instantaneous frequency and amplitude.
    This transform depicts the energy of a time-series signal over both time and frequency domains. By default, the
    output is aggregated over time and IMFs, giving only the frequency spectrum.
    """
    # Preliminary checks and setups. Add a dimension if necessary.
    instant_freq = instant_freq[:, np.newaxis] if instant_freq.ndim == 1 else instant_freq
    instant_amp = instant_amp[:, np.newaxis] if instant_amp.ndim == 1 else instant_amp

    bin_edges = _calculate_histogram_bins_and_centers(instant_freq.ravel())

    # Calculate the 2D Hilbert spectrum
    spectral_data = _compute_hilbert_huang_2d_spectrum(instant_freq, instant_amp, bin_edges)

    # Calculate the final spectrum as a power spectrum and return the result in the desired format (sparse or dense)
    final_spectra = spectral_data**2 if use_sparse_format else spectral_data.todense()
    return final_spectra


def hilbert_huang_transform(
    signal: pd.Series,
    sift_thresh: float = 1e-8,
    max_num_imfs: Optional[int] = None,
    error_tolerance: float = 0.05,
    return_trend: bool = True,
) -> pd.Series:
    r"""Perform the Hilbert-Huang Transform (HHT) to find the trend of a signal.

    The Hilbert-Huang Transform is a technique that combines Empirical Mode Decomposition (EMD)
    and the Hilbert Transform to analyze non-stationary and non-linear time series data,
    where the Hilbert transform is applied to each mode after having performed the EMD.
    Non-stationary signals are signals that vary in frequency and amplitude over time,
    and cannot be adequately represented by fixed parameters, whereas non-linear signals are signals
    that cannot be represented by a linear function and can exhibit complex and unpredictable behavior.
    Signals from different physical phenomena in the world are rarely purely linear or
    stationary, so the HHT is a useful tool for analyzing real-world signals.

    Given their complexity, it is often difficult to study the entire signals as a whole.
    Therefore, the HHT aims to capture the time-varying nature and non-linear dynamics of such signals by
    decomposing them into individual oscillatory components by the use the EMD.
    These components are oscillatory modes of the full signal with well-defined instantaneous frequencies called
    Intrinsic Mode Functions (IMFs). We calculate the hilbert spectrum of the IMFs to determine the significant ones
    from which we extract the trend as their sum. You can read more about this in the given sources.

    Args:
        signal (pd.Series): Input time series signal.
        sift_thresh (float, optional): The threshold used for sifting convergence. Defaults to 1e-8.
        max_num_imfs (int, optional): The maximum number of oscillatory components to extract. If None, extract all IMFs. Defaults to None.
        error_tolerance (float, optional): The tolerance for determining significant IMFs. Defaults to 0.05.
        return_trend (bool, optional): Flag indicating whether to return the trend component of the signal. Defaults to True.

    Returns:
        pd.Series: The detrended signal if `return_trend` is False, otherwise, the trend component of the signal.

    Raises:
        ValueError: If `sift_thresh` is not a number higher than zero.
        ValueError: If `max_num_imfs` is not an integer higher than zero.
        ValueError: If `error_tolerance` is not higher than zero.
        Any exceptions that may occur during Empirical Mode Decomposition (EMD).


    References:
        - Huang, Norden E., et al. "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis."
        Proceedings of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences 454.1971 (1998): 903-995.
        - Yang, Zhijing & Bingham, Chris & Ling, Bingo & Gallimore, Michael & Stewart, Paul & Zhang, Yu. (2012).
        "Trend extraction based on Hilbert-Huang transform." 1-5. 10.1109/CSNDSP.2012.6292713.
    """
    from PyEMD import EMD

    if sift_thresh <= 0:
        raise ValueError("The sifting threshold must be positive")
    if max_num_imfs is not None and max_num_imfs <= 0:
        raise ValueError("The maximum number of IMFs must be positive")
    if error_tolerance <= 0:
        raise ValueError("Error tolerance must be positive")

    signal_array = signal.to_numpy()

    # Empirical Mode Decomposition
    emd = EMD()
    # If max_num_imfs is None, Python's slicing will automatically include all IMFs.
    imfs = emd(signal_array)[:max_num_imfs]

    # Compute the analytic signal for each IMF using the Hilbert transform and store them in a list
    analytic_signals = [hilbert(imf) for imf in imfs]

    # Find the total number of IMFs and the index of the last IMF
    index_of_the_last_imf = imfs.shape[1] - 1
    significant_imf_index_rho = len(imfs) - 1

    # Compute the instantaneous frequency and amplitude of each IMF
    phase = np.unwrap(np.angle(analytic_signals))
    dt = np.mean(np.diff(get_timestamps(signal, "s").to_numpy()))
    frequency = np.gradient(phase) / (2 * np.pi * dt)
    amplitude = np.abs(analytic_signals)

    # Do some array gymnastics to get the right shapes
    frequency_array = np.array(frequency).T

    # average over time
    avg_frequency_array = np.mean(frequency_array, axis=-1)

    amplitude_array = np.array(amplitude)

    flat_amplitude = amplitude_array.flatten()
    flat_avg_frequency = avg_frequency_array.flatten()

    # compute Hilbert-Huang spectrum for each IMF separately and sum over time to get Hilbert marginal spectrum
    hht = _hilbert_huang_spectrum(flat_avg_frequency, flat_amplitude)

    # Loop for rho calculations
    for i in range(index_of_the_last_imf - 1, -1, -1):
        rho = np.sum(hht[:, i] * hht[:, i + 1]) / (np.sum(hht[:, i]) * np.sum(hht[:, i + 1]))  # type: ignore

        if rho < error_tolerance:
            break
        else:
            significant_imf_index_rho -= 1

    trend_rho = np.sum(imfs[significant_imf_index_rho:], axis=0, dtype=np.float64)
    trend_series_rho = pd.Series(trend_rho, index=signal.index)

    # Return the trend component of the signal if return_trend is True, otherwise, return the detrended signal
    result_rho = trend_series_rho if return_trend else signal - trend_series_rho

    return result_rho
