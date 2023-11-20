from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scipy.signal import hilbert

from indsl.ts_utils import get_timestamps


MIN_DATA_PT = 10

DROP_SENTINAL = np.iinfo(np.int32).min


class EMDSiftConvergenceError(Exception):
    """Exception

    Exception to be raised when the Empirical Mode Decomposition (EMD) sift process fails to converge.
    """

    def __init__(self, message: str):
        """Initializes the EMDSiftConvergenceError instance.

        Args:
            message (str): Message
                Description of the error explaining why the sift process failed to converge.
        """
        super().__init__(message)
        self.message = message


def bin_indices(data, boundaries):
    """Determine the bin indices

    Determine the bin index for each value based on bin edges.

    Args:
        data (array_like): Data
            Sequence of values to be categorized into bins
        boundaries (array_like): Boundaries
            Sequence defining the boundaries of the bins. Having N boundaries will define N-1 bins.
            Each boundary is inclusive at the start and exclusive at the end.

    Returns:
        ndarray
            An array denoting the index of the bin each data point belongs to.

    """
    # Identify values that are outside the boundaries
    out_of_bounds = np.logical_or(data < boundaries[0], data >= boundaries[-1])

    # Get bin indices for data
    indices = np.digitize(data, boundaries) - 1

    # Assign sentinel value to out-of-bound observations
    indices[out_of_bounds] = DROP_SENTINAL

    return indices


def compute_hilbert_huang_2d_spectrum(values_X, values_Z, x_boundaries):
    """Calculate a 2D Hilbert-Huang power distribution in sparse format.

    This utility function generates a sparse matrix that captures a two-dimensional
    power distribution. It's typically not intended for direct end-user interaction.

    Args:
        values_X (ndarray): Values X
            A 2D array with values for the primary dimension, typically [samples x imfs]
        values_Z (ndarray): Values Z
            A 2D array of amplitude or power corresponding to values_X
        x_boundaries (ndarray): X boundaries
            A vector defining bin boundaries for values_X

    Returns:
        sparse_spectrum (COO): Sparse spectrum
            A sparse representation of the 2D power distribution.

    """
    from sparse import COO

    # Ensure inputs are verified in calling functions.
    # Determine bin indices for primary dimension
    primary_indices = bin_indices(values_X, x_boundaries)

    # Establish indices for the temporal dimension and adjust shape to fit
    original_indices = np.arange(primary_indices.shape[0])[:, np.newaxis]
    time_indices = np.broadcast_to(original_indices, primary_indices.shape)

    # Identify indices for the IMF dimension and adjust shape accordingly
    original_imf_indices = np.arange(values_X.shape[1])[np.newaxis, :]
    imf_indices = np.broadcast_to(original_imf_indices, primary_indices.shape)

    # Generate COO coordinate matrix in a vectorized fashion
    coordinate_matrix = np.c_[primary_indices.flatten(), time_indices.flatten(), imf_indices.flatten()].T

    # Compute exclusions based on coordinates
    exclusions = np.any(coordinate_matrix == DROP_SENTINAL, axis=0)

    # Reshape exclusions based on its length
    reshaped_exclusions = exclusions.reshape(values_Z.shape)

    # Mask the Z-values based on the exclusions matrix
    values_Z_masked = np.ma.masked_array(values_Z, mask=reshaped_exclusions)
    values_Z_masked = values_Z_masked.squeeze()

    # Flatten the masked array to remove masked values
    pointwise_exclusions = reshaped_exclusions.squeeze()

    # Determine the valid points based on pointwise exclusions (i.e. any row-wise exclusions)
    valid_points = ~pointwise_exclusions

    # Use this to index the flattened Z-values and the coordinates
    adjusted_coordinate_matrix = coordinate_matrix[:, valid_points]

    # Determine the final matrix dimensions
    spectrum_shape = (x_boundaries.shape[0] - 1, primary_indices.shape[0], primary_indices.shape[1])

    sparse_spectrum = COO(adjusted_coordinate_matrix, values_Z_masked, shape=spectrum_shape)

    return sparse_spectrum


def validate_dimensionality(array_to_validate):
    """Confirm that all provided arrays have two dimensions.

    If an array is 1-dimensional, a second singleton dimension will be appended.
    Also, the function checks that arrays have consistent dimensions.
    """
    modified_arrays = []
    if array_to_validate.ndim == 1:
        modified_arrays = array_to_validate[:, np.newaxis]

    min_ndim = float("inf")
    if array_to_validate.ndim < min_ndim:
        min_ndim = array_to_validate.ndim
    else:
        raise ValueError("All arrays must have the same number of dimensions.")

    if len(modified_arrays) == 1:
        return modified_arrays[0]
    else:
        return modified_arrays


def finalize_spectrum_processing(
    input_spectrum,
    output_as_sparse=True,
):
    """Conduct standard post-processing on the provided spectrum.

    Args:
        input_spectrum (ndarray): Input spectrum
            Spectrum of 2 or 3 dimensions, typically sparse.
        output_as_sparse (bool, optional): Output as sparse
            Whether to return the output as a sparse array. Default is False.

    Returns:
        ndarray
            Post-processed spectrum.
    """
    input_spectrum = input_spectrum**2

    if not output_as_sparse:
        input_spectrum = input_spectrum.todense()
    else:
        if not sp.issparse(input_spectrum):
            input_spectrum = sp.csr_matrix(input_spectrum)

    return input_spectrum


def establish_histogram_bins(min_value, max_value, bin_count):
    """Determine the bin edges and central values for histogram construction.

    Args:
        min_value (float): Lower boundary for bin edges.
        max_value (float): Upper boundary for bin edges.
        bin_count (int): Total number of bins to create.

    Returns:
        tuple: A tuple containing:
            - bin_edges (ndarray): 1D array denoting bin edges.
            - bin_centers (ndarray): 1D array indicating bin central values.

    Examples:
        Creating histogram bins between 1 Hz and 5 Hz with four linearly spaced bins:
        >>> bin_edges, bin_centers = establish_histogram_bins(1, 5, 4)
        >>> print(bin_edges)
        [1. 2. 3. 4. 5.]
        >>> print(bin_centers)
        [1.5 2.5 3.5 4.5]
    """
    bin_edges = np.linspace(min_value, max_value, bin_count + 1)

    # Calculate the center values for the bins.
    bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)])

    return bin_edges, bin_centers


def calculate_histogram_bins(dataset, bin_count=None, margin=1e-3, bin_limit=2048):
    """Calculate histogram bins.

    Determine bin edges and centers for histogram.

    Args:
        dataset (ndarray): Data array that will influence the histogram bin settings.
        bin_count (int, optional): Explicit number of bins. If not specified, the count is inferred using the specified strategy.
        margin (float, optional): Small value added or subtracted to dataset's min and max values for bins. Defaults to 1e-3.
        bin_limit (int, optional): Maximum allowable number of bins. Defaults to 2048.

    Returns:
        tuple: A tuple containing:
            - edges (ndarray): 1D array of bin edges.
            - centers (ndarray): 1D array of bin centers.
    """
    data_lower_bound = dataset.min() - margin
    data_upper_bound = dataset.max() + margin

    if bin_count is None:
        bin_count = int(np.sqrt(dataset.size))

    # Ensure bin count doesn't exceed the allowed limit.
    bin_count = min(bin_count, bin_limit)

    return establish_histogram_bins(data_lower_bound, data_upper_bound, bin_count)


def calculate_bin_centers_from_edges(bin_edges):
    """Calculate bin centers from edges.

    The function calculates the bin centers from the given array of bin edges.
    """
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    return bin_centers


def adjust_histogram_bins(dataset):
    """Determine appropriate histogram bin settings.

    This function determines the appropriate histogram
    bin settings based on the provided dataset array and returns one array for
    the bin edges and one for the bin centers.
    """
    edges, centers = calculate_histogram_bins(dataset.ravel())

    return edges, centers


def hilbert_huang_spectrum(
    instant_freq,
    instant_amp,
    bin_edges=None,
    use_sparse_format=False,
):
    """Compute a Hilbert-Huang Spectrum (HHT).

    Calculate the Hilbert-Huang Transform (HHT) from instantaneous frequency and amplitude.

    This transform depicts the energy of a time-series signal over both time and frequency domains. By default, the
    output is aggregated over time and IMFs, giving only the frequency spectrum.

    Args:
        instant_freq (ndarray): Instant frequency
            2D array of instantaneous frequencies.
        instant_amp (ndarray): Instant amplitude
            2D array of instantaneous amplitudes.
        bin_edges (ndarray, tuple, None, optional): Bin edges
            Specifies the frequency bins for the spectrum:
                - ndarray: Directly defines bin edges.
                - tuple: Can be passed to generate histogram bin edges.
                - None: Automatically generates bins based on input data.
        use_sparse_format (bool, optional): Use sparse format
            Whether to use a sparse representation for the output.
            Recommended for large outputs. Default is False.

    Returns:
        frequency_bins (ndarray): Array of histogram bin centers for each frequency.
        final_spectra (ndarray): The calculated Hilbert-Huang Transform.

    Notes:
        The sparse output uses the COOrdinate format from the sparse package.
        This is memory-efficient, but might not be compatible with all functions expecting full arrays.

    References:
        Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert spectrum
        for nonlinear and non-stationary time series analysis. Proceedings of the Royal Society of London.
        Series A: Mathematical, Physical and Engineering Sciences, 454(1971), 903–995.
    """
    # Preliminary checks and setups
    instant_freq = validate_dimensionality(instant_freq)
    instant_amp = validate_dimensionality(instant_amp)

    bin_edges, frequency_bins = adjust_histogram_bins(instant_freq)

    # Compute the 2D spectrum
    spectral_data = compute_hilbert_huang_2d_spectrum(instant_freq, instant_amp, bin_edges)

    final_spectra = finalize_spectrum_processing(
        spectral_data,
        output_as_sparse=use_sparse_format,
    )

    return frequency_bins, final_spectra


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
    Intrinsic Mode Functions (IMFs).

    \textbf{Empirical Mode Decomposition} is a technique used to decompose a given signal into a set of
    intrinsic mode functions. To perform EMD, we start by finding the local maxima and minima in the signal.
    Then, we smoothly connect these points to create upper and lower envelopes using spline interpolation
    that capture the signal's overall trend. Next, we calculate the average of these envelopes and subtract
    it from the original signal, resulting in the first IMF — a high-frequency component. We repeat this process,
    sifting out the IMFs one by one until each IMF satisfies the two conditions for it to be an IMF. Read
    more about this in the given sources. The final result is a collection of IMFs, ordered by their frequencies,
    that represent the different oscillatory modes present in the original signal.

    \textbf{The Hilbert Transform} is a mathematical operation that, when applied to a real-valued time series signal,
    produces a corresponding analytic signal which is a complex-valued function derived
    from the initial signal, containing both the original signal and its Hilbert Transform.
    Given a real signal :math:`f(t)`, the Hilbert Transform :math:`H(f(t))` is defined in the full
    analytical signal :math:`z(t)` as

    .. math::
        z(t) = f(t) + i\{H(f(t))\} = f(t) + i\left(\frac{1}{\pi}P \int_{-\infty}^{\infty} \frac{f(\tau)}{t-\tau}d\tau\right).

    where :math:`P` is the Cauchy principal value to avoid singularities so that the integral converges.
    But why couple the EMD with the Hilbert transform? We complexify the signal through the Hilbert transform
    in order to access additional information about the signal such as the phase, i.e. the angle between the complex
    and real parts of the analytic signal, which gives us the instantaneous frequencies as the time derivative
    of the phase

    .. math::
        \omega(t) = \frac{d\phi(t)}{dt}.

    By analyzing the instantaneous frequency and amplitude of each IMF, the HHT provides a time-frequency
    representation of the signal, known as the Hilbert spectrum. The Hilbert spectrum shows how the
    frequency content of the signal changes over time, making it particularly useful for analyzing
    nonstationary signals with time-varying characteristics.

    After applying the HHT, the `hilbert_huang_transform` function returns the detrended signal if the `return_trend`
    parameter is set to False. This is obtained by subtracting the trend component from the original signal.
    The trend component represents the overall long-term behavior of the signal, capturing any gradual or systematic
    changes that are not part of the oscillatory components. On the other hand, if `return_trend` is set to True,
    the function returns the trend component itself. This allows users to analyze and study the trend separately
    from the oscillatory components, providing insights into the underlying patterns and dynamics of the signal.

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
        raise ValueError("The sifting threshold must be a number higher than zero")
    if max_num_imfs is not None and max_num_imfs <= 0:
        raise ValueError("The maximum number of oscillatory components must be an integer higher than zero")
    if error_tolerance <= 0:
        raise ValueError("The energy tolerance must be higher than zero")

    signal_array = signal.to_numpy()

    emd = EMD()
    imfs = emd(signal_array)  # Empirical Mode Decomposition

    # Truncate the IMFs to the max number of IMFs
    if max_num_imfs is not None and max_num_imfs <= len(imfs):
        imfs = imfs[:max_num_imfs]

    # Compute the analytic signal for each IMF using the Hilbert transform and store them in a list
    analytic_signals = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        analytic_signals.append(analytic_signal)

    # Find the total number of IMFs and the index of the last IMF
    total_number_of_imf: int = imfs.shape[1]
    index_of_the_last_imf: int = total_number_of_imf - 1
    significant_imf_index_rho = len(imfs) - 1

    # Compute the instantaneous frequency and amplitude of each IMF
    phase = np.unwrap(np.angle(analytic_signals))
    dt = np.mean(np.diff(get_timestamps(signal, "s").to_numpy()))
    frequency = np.gradient(phase) / (2 * np.pi * dt)
    amplitude = np.abs(analytic_signals)

    # Do some array gymnastics to get the right shapes
    frequency_array = np.array(frequency).T  # transpose to get the right shape
    avg_frequency_array = np.mean(frequency_array, axis=-1)  # average over time
    amplitude_array = np.array(amplitude)  # transpose to get the right shape

    flat_amplitude = amplitude_array.flatten()
    flat_avg_frequency = avg_frequency_array.flatten()

    assert flat_amplitude.shape == flat_avg_frequency.shape, "Mismatch between frequency and amplitude arrays"

    # compute Hilbert-Huang spectrum for each IMF separately and sum over time to get Hilbert marginal spectrum
    _, hht = hilbert_huang_spectrum(flat_avg_frequency, flat_amplitude)

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
