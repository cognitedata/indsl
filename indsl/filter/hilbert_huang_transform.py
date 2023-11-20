from typing import Optional, Tuple

import numpy as np
import pandas as pd

import emd
import logging

from PyEMD import EMD
from scipy.signal import hilbert  # type: ignore
from indsl.ts_utils import get_timestamps
from scipy.signal._peak_finding import peak_prominences
from scipy.signal import argrelextrema, find_peaks
from scipy import interpolate as interp
from scipy.interpolate import CubicSpline
import scipy.sparse as sp

MIN_DATA_PT = 10

DROP_SENTINAL = np.iinfo(np.int32).min
logger = logging.getLogger(__name__)


class EMDSiftConvergenceError(Exception):
    """
    Exception to be raised when the Empirical Mode Decomposition (EMD) sift process fails to converge.

    Attributes
    ----------
    message : str
        Description of the error explaining why the sift process failed to converge.
    """

    def __init__(self, message: str):
        """
        Initializes the EMDSiftConvergenceError instance.

        Parameters
        ----------
        message : str
            Description of the error explaining why the sift process failed to converge.
        """
        super().__init__(message)
        self.message = message
        logger.error("EMD Sift Convergence Error: %s", self.message)


def bin_indices(data, boundaries):
    """Determine the bin index for each value based on bin edges.

    Parameters
    ----------
    data : array_like
        Sequence of values to be categorized into bins
    boundaries : array_like
        Sequence defining the boundaries of the bins. Having N boundaries will define N-1 bins.
        Each boundary is inclusive at the start and exclusive at the end.

    Returns
    -------
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

    Parameters
    ----------
    values_X : ndarray
        A 2D array with values for the primary dimension, typically [samples x imfs]
    values_Z : ndarray
        A 2D array of amplitude or power corresponding to values_X
    x_boundaries : ndarray
        A vector defining bin boundaries for values_X

    Returns
    -------
    sparse_spectrum
        A sparse representation of the 2D power distribution.

    Notes
    -----
    Refer to the main function `hilberthuang` for more context.

    """
    # Ensure inputs are verified in calling functions.
    # Determine bin indices for primary dimension
    primary_indices = bin_indices(values_X, x_boundaries)

    # Establish indices for the temporal dimension and adjust shape to fit
    original_indices = np.arange(primary_indices.shape[0])[:, np.newaxis]

    time_indices = np.broadcast_to(original_indices, primary_indices.shape)

    # time_indices = np.broadcast_to(np.arange(primary_indices.shape[0])[:, np.newaxis], primary_indices.shape)
    # Identify indices for the IMF dimension and adjust shape accordingly
    original_imf_indices = np.arange(values_X.shape[1])[np.newaxis, :]
    imf_indices = np.broadcast_to(original_imf_indices, primary_indices.shape)
    # imf_indices = np.broadcast_to(np.arange(values_X.shape[1])[np.newaxis, :], primary_indices.shape)

    print("SHAPE OF primary_indices:", primary_indices.shape)
    print("SHAPE OF original_indices:", original_indices.shape)
    print("SHAPE OF time_indices:", time_indices.shape)
    print("SHAPE OF imf_indices:", imf_indices.shape)

    # # Generate COO coordinate matrix in a vectorized fashion
    # coordinate_matrix = np.c_[primary_indices.flatten(), time_indices.flatten(), imf_indices.flatten()].T

    # # Adjust amplitude values to align with coordinates
    # values_Z = values_Z.flatten()
    # flattened_Z = values_Z.flatten()

    # original_shape_Z = values_Z.shape  # (n_samples, n_imfs)

    # # Exclude data outside the given bin boundaries
    # exclusions = np.any(coordinate_matrix == DROP_SENTINAL, axis=0)
    # print("SHAPE OF EXCLUSIONS: ", exclusions.shape)
    # print("COORDINATE MATRIX SHAPE: ", coordinate_matrix.shape)
    # print("VALUES Z SHAPE: ", values_Z.shape)
    # # coordinate_matrix = np.delete(coordinate_matrix, exclusions, axis=1)
    # # values_Z = np.delete(values_Z, exclusions)

    # adjusted_coordinate_matrix = np.delete(coordinate_matrix, exclusions, axis=1)
    # adjusted_Z = np.delete(flattened_Z, exclusions)

    # # Generate COO coordinate matrix in a vectorized fashion
    coordinate_matrix = np.c_[primary_indices.flatten(), time_indices.flatten(), imf_indices.flatten()].T

    # # Compute exclusions before flattening values_Z
    # exclusions = np.any(coordinate_matrix == DROP_SENTINAL, axis=0)
    # print("SHAPE OF EXCLUSIONS: ", exclusions.shape)

    # # Flatten after computing the exclusions
    # flattened_Z = values_Z.flatten()
    # print("VALUES Z SHAPE: ", flattened_Z.shape)

    # # Compute exclusions based on coordinates for every entry in flattened_Z
    # # This is needed since every value in flattened_Z has 3 coordinates.
    # exclusion_pairs = exclusions.reshape(-1, 3)
    # valid_values = ~np.any(exclusion_pairs, axis=1)

    # # Filter out values and coordinates based on valid_values
    # adjusted_Z = flattened_Z[valid_values]
    # adjusted_coordinate_matrix = coordinate_matrix[:, valid_values.flatten()]

    # # Use exclusions on flattened_Z
    # adjusted_Z = np.delete(flattened_Z, np.where(exclusions))

    # # Use exclusions on coordinate_matrix
    # adjusted_coordinate_matrix = np.delete(coordinate_matrix, np.where(exclusions), axis=1)

    # # Determine the final matrix dimensions
    # spectrum_shape = (x_boundaries.shape[0] - 1, primary_indices.shape[0], primary_indices.shape[1])

    # # Construct the sparse spectrum representation
    # from sparse import COO

    # # sparse_spectrum = COO(coordinate_matrix, values_Z, shape=spectrum_shape)
    # sparse_spectrum = COO(adjusted_coordinate_matrix, adjusted_Z, shape=spectrum_shape)

    # return sparse_spectrum

    # Generate COO coordinate matrix in a vectorized fashion
    #    coordinate_matrix = np.c_[primary_indices.flatten(), time_indices.flatten(), imf_indices.flatten()].T
    # coordinate_matrix = np.vstack([primary_indices.flatten(), time_indices.flatten()])

    # Compute exclusions based on coordinates
    exclusions = np.any(coordinate_matrix == DROP_SENTINAL, axis=0)
    print("SHAPE OF EXCLUSIONS: ", exclusions.shape)

    # Flatten Z values
    flattened_Z = values_Z.flatten()
    print("VALUES Z SHAPE: ", flattened_Z.shape)

    # Instead of reshaping exclusions into groups of 3, let's break it down into groups of 2
    # This is because we've constructed the coordinate matrix from 2D indices.
    # exclusion_pairs = exclusions.reshape(-1, 2)
    # valid_values = ~np.any(exclusion_pairs, axis=1)

    # Reshape exclusions based on its length
    reshaped_exclusions = exclusions.reshape(values_Z.shape)
    print("SHAPE OF RESHAPED EXCLUSIONS:", reshaped_exclusions.shape)

    values_Z_masked = np.ma.masked_array(values_Z, mask=reshaped_exclusions)
    values_Z_masked = values_Z_masked.squeeze()
    print("Shape of values_Z_masked:", values_Z_masked.shape)

    # Check any row-wise exclusions
    # pointwise_exclusions = np.any(reshaped_exclusions, axis=0)
    pointwise_exclusions = reshaped_exclusions.squeeze()
    print("Sum of pointwise exclusions:", np.sum(pointwise_exclusions))

    # Now, pointwise_exclusions should have shape (14976,)

    valid_points = ~pointwise_exclusions
    print("Sum of valid points:", np.sum(valid_points))

    # print("SHAPE OF RESHAPED_EXCLUSIONS:", reshaped_exclusions.shape)
    # print("SHAPE OF POINTWISE_EXCLUSIONS:", pointwise_exclusions.shape)
    # print("SHAPE OF VALID_POINTS:", valid_points.shape)

    # Use this to index the flattened Z-values and the coordinates
    adjusted_Z = flattened_Z[valid_points]
    adjusted_coordinate_matrix = coordinate_matrix[:, valid_points]

    # Filter out values and coordinates based on valid_values
    # adjusted_Z = flattened_Z[valid_values]
    # expanded_valid_values = np.tile(valid_values, (3, 1))
    # adjusted_coordinate_matrix = coordinate_matrix[:, expanded_valid_values[0]]

    # Determine the final matrix dimensions
    spectrum_shape = (x_boundaries.shape[0] - 1, primary_indices.shape[0], primary_indices.shape[1])

    # Construct the sparse spectrum representation
    from sparse import COO

    sparse_spectrum = COO(adjusted_coordinate_matrix, values_Z_masked, shape=spectrum_shape)

    return sparse_spectrum


def validate_dimensionality(arrays_to_validate, variable_names, calling_function):
    """
    Confirm that all provided arrays have two dimensions.

    If an array is 1-dimensional, a second singleton dimension will be appended.

    Parameters
    ----------
    arrays_to_validate : list of arrays
        Arrays to validate for having two dimensions.
    variable_names : list
        Corresponding names of the arrays in arrays_to_validate for clearer logging.
    calling_function : str
        Name of the function that invokes this validation function.

    Returns
    -------
    modified_arrays
        List of arrays after ensuring they have 2 dimensions.

    """
    modified_arrays = list(arrays_to_validate)
    for i, array in enumerate(arrays_to_validate):
        if array.ndim == 1:
            log_message = "In {0}: Adjusting dimensionality for '{1}'"
            logger.debug(log_message.format(calling_function, variable_names[i]))
            modified_arrays[i] = array[:, np.newaxis]

    if len(modified_arrays) == 1:
        return modified_arrays[0]
    else:
        return modified_arrays


def validate_dimensions_consistency(arrays_list, variable_labels, invoking_function, specific_axis=None):
    """
    Validate that all provided arrays have consistent dimensions.

    If dimensions aren't consistent across arrays, it will raise an error.

    Parameters
    ----------
    arrays_list : list of arrays
        Arrays whose dimensions are to be validated for consistency.
    variable_labels : list
        Corresponding labels for the arrays in arrays_list for better logging.
    invoking_function : str
        The function name that calls this validation function.
    specific_axis : int, optional
        Specific axis to check for dimensional consistency.
        If not provided, all axes will be compared.

    Raises
    ------
    ValueError
        If the arrays in arrays_list have inconsistent shapes.

    """
    # if specific_axis is None:
    #     axes_to_check = np.arange(arrays_list[0].ndim)
    # else:
    #     axes_to_check = [specific_axis]

    # Determine the minimum number of dimensions across all arrays
    min_ndim = float("inf")
    for arr in arrays_list:
        if arr.ndim < min_ndim:
            min_ndim = arr.ndim

    # Determine which axes to check
    if specific_axis is None:
        axes_to_check = list(np.arange(min_ndim).astype(int))
    else:
        if specific_axis < min_ndim:  # Check if the specific axis is within the bounds of the arrays
            axes_to_check = [specific_axis]  # Convert to list for consistency
        else:
            raise ValueError(
                f"The specific_axis {specific_axis} is out of bounds for arrays with dimensions less than {min_ndim}."
            )

    # for arr in arrays_list: # Check if the specific axis is within the bounds of the arrays
    #     print(arr.shape)
    # print(axes_to_check)

    dimensions_list = [
        tuple(np.array(arr.shape)[axes_to_check]) for arr in arrays_list
    ]  # Get the dimensions of the arrays along the specified axes
    is_consistent = [
        dimensions_list[0] == dim for dim in dimensions_list
    ]  # Check if the dimensions are consistent across all arrays

    if not all(is_consistent):
        log_msg = "In {0}: Dimensions inconsistency detected among inputs".format(invoking_function)
        logger.error(log_msg)

        error_details = "Differing dimensions among inputs: "
        for idx, arr in enumerate(arrays_list):
            error_details += "'{0}': {1}, ".format(variable_labels[idx], arr.shape)
        logger.error(error_details)
        raise ValueError(error_details)


def finalize_spectrum_processing(
    input_spectrum,
    dimensions_to_sum=None,
    output_mode="power",
    spectrum_scaling=None,
    temporal_axis=1,
    sampling_frequency=1,
    output_as_sparse=True,
    dense_output_size_limit=50,
):
    """
    Conduct standard post-processing on the provided spectrum.

    Parameters
    ----------
    input_spectrum : ndarray
        Spectrum of 2 or 3 dimensions, typically sparse.
    dimensions_to_sum : int or list of int, optional
        Indicate dimensions to sum across. Default is None.
    output_mode : {'power', 'amplitude'}, optional
        Whether to output in power (squared amplitude) or amplitude format. Default is 'power'.
    spectrum_scaling : {'density', 'spectrum', None}, optional
        The type of scaling to apply to the spectrum. Default is None.
    temporal_axis : int, optional
        The index of the time dimension for applying scaling. Default is 1.
    output_as_sparse : bool, optional
        Whether to return the output as a sparse array. Default is False.
    dense_output_size_limit : float, optional
        Max size (in GB) of dense output. If exceeded, raises an error. Default is 10 GB.

    Returns
    -------
    ndarray
        Post-processed spectrum.

    Notes
    -----
    Assumes input data have been pre-processed by upper-level functions.
    """

    if output_mode == "power":
        logger.debug("Transforming amplitude to power format")
        input_spectrum = input_spectrum**2

    if spectrum_scaling == "density":
        logger.debug("Applying 'density' scaling.")
        input_spectrum = input_spectrum / (sampling_frequency * input_spectrum.shape[temporal_axis])
    elif spectrum_scaling == "spectrum":
        logger.debug("Applying 'spectrum' scaling.")
        input_spectrum = input_spectrum / input_spectrum.shape[temporal_axis]
    elif spectrum_scaling:
        logger.error(f"Unrecognized scaling: {spectrum_scaling}")
        raise ValueError(f"Unrecognized scaling: {spectrum_scaling}")

    if dimensions_to_sum:
        original_shape = input_spectrum.shape
        input_spectrum = input_spectrum.sum(axis=dimensions_to_sum)
        logger.debug(
            f"Aggregating over dimensions {dimensions_to_sum}. Original shape {original_shape} -> New shape {input_spectrum.shape}"
        )

    byte_estimate = input_spectrum.size * 8
    gb_estimate = byte_estimate / (1024**3)

    if not output_as_sparse and dense_output_size_limit:
        if gb_estimate > dense_output_size_limit:
            err_msg = (
                f"Converting to dense will result in a {gb_estimate} Gb array, "
                f"exceeding the {dense_output_size_limit} Gb limit. Consider using a sparse array or increasing the limit."
            )
            logger.warning(err_msg)
            # raise RuntimeError(err_msg)
        input_spectrum = input_spectrum.todense()
        logger.debug(f"Output transformed to dense format, estimated at {gb_estimate}Gb")
    else:
        logger.debug(f"Output remains in sparse format, estimated at {gb_estimate}Gb")
        if not sp.issparse(input_spectrum):
            input_spectrum = sp.csr_matrix(input_spectrum)

    return input_spectrum


def establish_histogram_bins(min_value, max_value, bin_count, bin_spacing="linear"):
    """
    Determine the bin edges and central values for histogram construction.

    Parameters
    ----------
    min_value : float
        Lower boundary for bin edges.
    max_value : float
        Upper boundary for bin edges.
    bin_count : int
        Total number of bins to create.
    bin_spacing : {'linear', 'log'}, default='linear'
        Method to define spacing between bins, either linearly or logarithmically.

    Returns
    -------
    bin_edges : ndarray
        1D array denoting bin edges.
    bin_centers : ndarray
        1D array indicating bin central values.

    Examples
    --------
    Creating histogram bins between 1 Hz and 5 Hz with four linearly spaced bins:

    >>> bin_edges, bin_centers = establish_histogram_bins(1, 5, 4)
    >>> print(bin_edges)
    [1. 2. 3. 4. 5.]
    >>> print(bin_centers)
    [1.5 2.5 3.5 4.5]

    """
    if bin_spacing == "log":
        log_bounds = np.log([min_value, max_value])
        bin_edges = np.linspace(log_bounds[0], log_bounds[1], bin_count + 1)
        bin_edges = np.exp(bin_edges)
    elif bin_spacing == "linear":
        bin_edges = np.linspace(min_value, max_value, bin_count + 1)
    else:
        raise ValueError(f"Bin spacing '{bin_spacing}' is not valid. Choose 'log' or 'linear'.")

    # Calculate the center values for the bins.
    bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)])

    return bin_edges, bin_centers


def calculate_histogram_bins(
    dataset, bin_count=None, strategy="sqrt", axis_scale="linear", margin=1e-3, bin_limit=2048
):
    """
    Determine bin edges and centers for histogram based on dataset properties.

    If bin_count is given, strategy is disregarded.

    Parameters
    ----------
    dataset : ndarray
        Data array that will influence the histogram bin settings.
    bin_count : int, optional
        Explicit number of bins. If not specified, the count is inferred using the specified strategy.
    strategy : {'sqrt'}, default='sqrt'
        Approach to decide number of bins if bin_count isn't provided.
    axis_scale : {'linear', 'log'}, default='linear'
        Determines the scale for histogram axis.
    margin : float, default=1e-3
        Small value added or subtracted to dataset's min and max values for bins.
    bin_limit : int, default=2048
        Maximum allowable number of bins.

    Returns
    -------
    edges : ndarray
        1D array of bin edges.
    centers : ndarray
        1D array of bin centers.

    """
    data_lower_bound = dataset.min() - margin
    data_upper_bound = dataset.max() + margin

    if bin_count is None:
        if strategy == "sqrt":
            bin_count = int(np.sqrt(dataset.size))
        else:
            raise ValueError(f"Strategy '{strategy}' is not recognized. Consider using 'sqrt'.")

    # Ensure bin count doesn't exceed the allowed limit.
    bin_count = min(bin_count, bin_limit)

    return establish_histogram_bins(data_lower_bound, data_upper_bound, bin_count, axis_scale)


def calculate_bin_centers_from_edges(bin_edges, calculation_mode="mean"):
    """
    Derive bin centers from a given array of bin edges.

    Parameters
    ----------
    bin_edges : ndarray
        Array specifying bin edges.
    calculation_mode : str, {'mean', 'geometric'}, default='mean'
        Method to determine the bin centers. Options are:
        - 'mean': Average of two consecutive edges.
        - 'geometric': Geometric mean of two consecutive edges.

    Returns
    -------
    bin_centers : ndarray
        Calculated bin centers.

    Raises
    ------
    ValueError
        If the provided calculation_mode is not recognized.
    """

    if calculation_mode == "geometric":
        bin_centers = np.sqrt(bin_edges[1:] * bin_edges[:-1])
    elif calculation_mode == "mean":
        bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    else:
        raise ValueError(f"Unknown calculation mode '{calculation_mode}'. Options are 'mean' or 'geometric'.")

    return bin_centers


def adjust_histogram_bins(bin_params, dataset):
    """
    Determine the appropriate histogram bin settings based on user input.

    Parameters
    ----------
    bin_params : None or tuple(start, stop, nsteps) or np.ndarray
        User-provided parameters. If:
        - None: bins are automatically determined from the dataset.
        - Tuple of size three: bins are generated based on the given range and step size.
        - numpy.ndarray: input directly specifies bin edges, from which centers will be calculated.
    dataset : ndarray, optional
        Data array used for bin calculation if bin_params is None.

    Returns
    -------
    ndarray
        Array of bin edges.
    ndarray
        Array of bin centers.

    """
    if bin_params is None:
        # User hasn't provided specific parameters - derive bins from dataset.
        edges, centers = calculate_histogram_bins(dataset.ravel())
    elif isinstance(bin_params, tuple) and len(bin_params) in [3, 4]:
        # User specified bin range and steps - generate bins accordingly.
        edges, centers = establish_histogram_bins(*bin_params)  # type: ignore
    elif isinstance(bin_params, (list, tuple, np.ndarray)):
        # User provided direct bin edges - utilize them.
        edges = np.array(bin_params)
        centers = calculate_bin_centers_from_edges(edges)
    else:
        raise ValueError("Unrecognized input format for bin parameters.")

    return edges, centers


def hilbert_huang_spectrum(
    instant_freq,
    instant_amp,
    bin_edges=None,
    aggregate_time=True,
    aggregate_imfs=True,
    aggregation_mode="power",
    data_sample_rate=1,
    spectrum_scaling=None,
    use_sparse_format=False,
    size_limit_gb=10,
):
    """Compute a Hilbert-Huang Spectrum (HHT).

    Calculate the Hilbert-Huang Transform (HHT) from instantaneous frequency and amplitude.

    This transform depicts the energy of a time-series signal over both time and frequency domains. By default, the
    output is aggregated over time and IMFs, giving only the frequency spectrum. This behavior can be adjusted with
    the `aggregate_time` and `aggregate_imfs` arguments.

    Parameters
    ----------
    instant_freq : ndarray
        2D array of instantaneous frequencies.
    instant_amp : ndarray
        2D array of instantaneous amplitudes.
    bin_edges : {ndarray, tuple, None}, optional
        Specifies the frequency bins for the spectrum:
        - ndarray: Directly defines bin edges.
        - tuple: Can be passed to generate histogram bin edges.
        - None: Automatically generates bins based on input data.
    aggregate_time : bool, optional
        Whether to aggregate over the time dimension. Default is True.
    aggregate_imfs : bool, optional
        Whether to aggregate over the IMF dimension. Default is True.
    aggregation_mode : {'power','amplitude'}, optional
        Decides if the power or amplitudes should be aggregated. Default is 'power'.
    data_sample_rate : float, optional
        Sampling rate of the provided data. Default is 1.
    spectrum_scaling : {'density', 'spectrum', None}, optional
        Determines the normalization or scaling applied to the spectrum.
    use_sparse_format : bool, optional
        Whether to use a sparse representation for the output. Recommended for large outputs. Default is False.
    size_limit_gb : float, optional
        If the non-sparse output exceeds this size (in GB), an error is raised. Default is 10 GB.

    Returns
    -------
    frequency_bins : ndarray
        Array of histogram bin centers for each frequency.
    final_spectra : ndarray
        The calculated Hilbert-Huang Transform.

    Notes
    -----
    The sparse output uses the COOrdinate format from the sparse package. This is memory-efficient, but might not be
    compatible with all functions expecting full arrays.

    References
    ----------
    Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert spectrum for nonlinear and
    non-stationary time series analysis. Proceedings of the Royal Society of London. Series A: Mathematical, Physical
    and Engineering Sciences, 454(1971), 903–995.
    """
    # Preliminary checks and setups
    instant_freq, instant_amp = validate_dimensionality(
        [instant_freq, instant_amp], ["instant_freq", "instant_amp"], "hilbert_huang_transform"
    )
    validate_dimensions_consistency(
        (instant_freq, instant_amp), ("instant_freq", "instant_amp"), "hilbert_huang_transform"
    )

    logger.info("INITIATED: Hilbert-Huang Transformation calculation")
    logger.debug("Processing on {0} samples across {1} IMFs ".format(instant_freq.shape[0], instant_freq.shape[1]))
    bin_edges, frequency_bins = adjust_histogram_bins(bin_edges, instant_freq.flatten())
    logger.debug("Frequency bins: {0} to {1} with {2} divisions".format(bin_edges[0], bin_edges[-1], len(bin_edges)))

    # Compute the 2D spectrum
    print(
        "Shape of instant_amp and instant_freq, respectively, before compute_hilbert_huang_2d_spectrum:",
        instant_amp.shape,
        instant_freq.shape,
    )

    spectral_data = compute_hilbert_huang_2d_spectrum(instant_freq, instant_amp, bin_edges)

    print("Shape of spectral_data after compute_hilbert_huang_2d_spectrum:", spectral_data.shape)

    dimension_to_aggregate = np.where([0, aggregate_time, aggregate_imfs])[0]

    final_spectra = finalize_spectrum_processing(
        spectral_data,
        dimensions_to_sum=dimension_to_aggregate,
        output_mode=aggregation_mode,
        spectrum_scaling=spectrum_scaling,
        temporal_axis=1,
        sampling_frequency=data_sample_rate,
        output_as_sparse=use_sparse_format,
        dense_output_size_limit=size_limit_gb,
    )

    logger.info("FINISHED: Hilbert-Huang Transformation - resulting size {0}".format(final_spectra.shape))
    return frequency_bins, final_spectra


def compute_frequency_attributes(imfs, sampling_rate, calculation_method, freq_smoothing=3, phase_smoothing=5):
    """
    Determine instantaneous phase, frequency, and amplitude for a collection of IMFs.

    This function utilizes various methods from references [1] and [2].

    Parameters
    ----------
    imfs : ndarray
        Collection of Input Mode Functions (IMFs).
    sampling_rate : float
        The rate at which the signal is sampled, in Hz.
    calculation_method : {'hilbert', 'quad', 'direct_quad', 'nht'}
        Technique to be used for deriving frequency characteristics.
    freq_smoothing : int, optional
        Window length for smoothing frequency, default is 3.
    phase_smoothing : int, optional
        Window length to smooth the unwrapped phase, default is 5.

    Returns
    -------
    instantaneous_phase : ndarray
        Calculated instantaneous phase estimates.
    instantaneous_frequency : ndarray
        Estimated instantaneous frequencies.
    instantaneous_amplitude : ndarray
        Computed instantaneous amplitude estimates.

    References
    ----------
    .. [1] Huang, N. E., et al. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proceedings
       of the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 454(1971), 903–995.
       https://doi.org/10.1098/rspa.1998.0193
    .. [2] Huang, N. E., et al. (2009). On Instantaneous Frequency. Advances in Adaptive Data Analysis,
       1(2), 177–229. https://doi.org/10.1142/s1793536909000096
    """
    logger.info("INITIATED: Frequency attribute computation")

    imfs = validate_dimensionality(imfs, ["imf"], "compute_frequency_attributes")
    logger.debug(f"Processing {0} samples across {1} IMFs at a sampling rate of {sampling_rate}")


def ensure_1d_with_singleton(to_check, names, func_name):
    """Check that a set of arrays are all vectors with singleton second dimensions.

    1d arrays will have a singleton second dimension added and an error will be
    raised for non-singleton 2d or greater than 2d inputs.

    Parameters
    ----------
    to_check : list of arrays
        List of arrays to check for equal dimensions
    names : list
        List of variable names for arrays in to_check
    func_name : str
        Name of function calling ensure_equal_dims

    Returns
    -------
    out
        Copy of arrays in to_check with '1d with singleton' shape.

    Raises
    ------
    ValueError
        If any input is a 2d or greater array

    """
    out_args = list(to_check)
    for idx, xx in enumerate(to_check):
        if (xx.ndim >= 2) and np.all(xx.shape[1:] == np.ones_like(xx.shape[1:])):
            # nd input where all trailing are ones
            msg = "Checking {0} inputs - Trimming trailing singletons from input '{1}' (input size {2})"
            logger.debug(msg.format(func_name, names[idx], xx.shape))
            out_args[idx] = np.squeeze(xx)[:, np.newaxis]
        elif (xx.ndim >= 2) and np.all(xx.shape[1:] == np.ones_like(xx.shape[1:])) == False:  # noqa: E712
            # nd input where some trailing are not one
            msg = "Checking {0} inputs - trailing dims of input '{1}' {2} must be singletons (length=1)"
            logger.error(msg.format(func_name, names[idx], xx.shape))
            raise ValueError(msg)
        elif xx.ndim == 1:
            # Vector input - add a dummy dimension
            msg = "Checking {0} inputs - Adding dummy dimension to input '{1}'"
            logger.debug(msg.format(func_name, names[idx]))
            out_args[idx] = out_args[idx][:, np.newaxis]

    if len(out_args) == 1:
        return out_args[0]
    else:
        return out_args


def compute_parabolic_extrema(y, locs):
    """Compute a parabolic refinement extrema locations.

    Parabolic refinement is computed from in triplets of points based on the
    method described in section 3.2.1 from Rato 2008 [1]_.

    Parameters
    ----------
    y : array_like
        A [3 x nextrema] array containing the points immediately around the
        extrema in a time-series.
    locs : array_like
        A [nextrema] length vector containing x-axis positions of the extrema

    Returns
    -------
    numpy array
        The estimated y-axis values of the interpolated extrema
    numpy array
        The estimated x-axis values of the interpolated extrema

    References
    ----------
    .. [1] Rato, R. T., Ortigueira, M. D., & Batista, A. G. (2008). On the HHT,
    its problems, and some solutions. Mechanical Systems and Signal Processing,
    22(6), 1374–1394. https://doi.org/10.1016/j.ymssp.2007.11.028

    """
    # Parabola equation parameters for computing y from parameters a, b and c
    # w = np.array([[1, 1, 1], [4, 2, 1], [9, 3, 1]])
    # ... and its inverse for computing a, b and c from y
    w_inv = np.array([[0.5, -1, 0.5], [-5 / 2, 4, -3 / 2], [3, -3, 1]])
    abc = w_inv.dot(y)

    # Find co-ordinates of extrema from parameters abc
    tp = -abc[1, :] / (2 * abc[0, :])
    t = tp - 2 + locs
    y_hat = tp * abc[1, :] / 2 + abc[2, :]

    return t, y_hat


def _find_extrema(x, peak_prom_thresh=None, parabolic_extrema=False):
    """Identify extrema within a time-course.

    This function detects extrema using a scipy.signals.argrelextrema. Extrema
    locations can be refined by parabolic intpolation and optionally
    thresholded by peak prominence.

    Parameters
    ----------
    x : ndarray
       Input signal
    peak_prom_thresh : {None, float}
       Only include peaks which have prominences above this threshold or None
       for no threshold (default is no threshold)
    parabolic_extrema : bool
        Flag indicating whether peak estimation should be refined by parabolic
        interpolation (default is False)

    Returns
    -------
    locs : ndarray
        Location of extrema in samples
    extrema : ndarray
        Value of each extrema

    """
    from scipy.signal import argrelextrema

    ext_locs = argrelextrema(x, np.greater, order=1)[0]

    if len(ext_locs) == 0:
        return np.array([]), np.array([])

    from scipy.signal._peak_finding import peak_prominences

    if peak_prom_thresh is not None:
        prom, _, _ = peak_prominences(x, ext_locs, wlen=3)
        keeps = np.where(prom > peak_prom_thresh)[0]
        ext_locs = ext_locs[keeps]

    if parabolic_extrema:
        y = np.c_[x[ext_locs - 1], x[ext_locs], x[ext_locs + 1]].T
        ext_locs, max_pks = compute_parabolic_extrema(y, ext_locs)
        return ext_locs, max_pks
    else:
        return ext_locs, x[ext_locs]


def _pad_extrema_numpy(locs, mags, lenx, pad_width, loc_pad_opts, mag_pad_opts):
    """Pad extrema using a direct call to np.pad.

    Extra paddings are carried out if the padded values do not span the whole
    range of the original time-series (defined by lenx)

    Parameters
    ----------
    locs : ndarray
        location of extrema in time
    mags : ndarray
        magnitude of each extrema
    lenx : int
        length of the time-series from which locs and mags were identified
    pad_width : int
        number of extra extrema to pad
    loc_pad_opts : dict
        dictionary of argumnents passed to np.pad to generate new extrema locations
    mag_pad_opts : dict
        dictionary of argumnents passed to np.pad to generate new extrema magnitudes

    Returns
    -------
    ndarray
        location of all extrema (including padded and original points) in time
    ndarray
        magnitude of each extrema (including padded and original points)

    """
    logger.verbose("Padding {0} extrema in signal X {1} using method '{2}'".format(pad_width, lenx, "numpypad"))

    if not loc_pad_opts:  # Empty dict evaluates to False
        loc_pad_opts = {"mode": "reflect", "reflect_type": "odd"}
    else:
        loc_pad_opts = loc_pad_opts.copy()  # Don't work in place...
    loc_pad_mode = loc_pad_opts.pop("mode")

    if not mag_pad_opts:  # Empty dict evaluates to False
        mag_pad_opts = {"mode": "median", "stat_length": 1}
    else:
        mag_pad_opts = mag_pad_opts.copy()  # Don't work in place...
    mag_pad_mode = mag_pad_opts.pop("mode")

    # Determine how much padding to use
    if locs.size < pad_width:
        pad_width = locs.size

    # Return now if we're not padding
    if (pad_width is None) or (pad_width == 0):
        return locs, mags

    # Pad peak locations
    ret_locs = np.pad(locs, pad_width, loc_pad_mode, **loc_pad_opts)

    # Pad peak magnitudes
    ret_mag = np.pad(mags, pad_width, mag_pad_mode, **mag_pad_opts)

    # Keep padding if the locations don't stretch to the edge
    count = 0
    while np.max(ret_locs) < lenx or np.min(ret_locs) >= 0:
        logger.debug("Padding again - first ext {0}, last ext {1}".format(np.min(ret_locs), np.max(ret_locs)))
        logger.debug(ret_locs)
        ret_locs = np.pad(ret_locs, pad_width, loc_pad_mode, **loc_pad_opts)
        ret_mag = np.pad(ret_mag, pad_width, mag_pad_mode, **mag_pad_opts)
        count += 1
        # if count > 5:
        #    raise ValueError

    return ret_locs, ret_mag


def _pad_extrema_rilling(indmin, indmax, X, pad_width):
    """Pad extrema using the method from Rilling.

    This is based on original matlab code in boundary_conditions_emd.m
    downloaded from: https://perso.ens-lyon.fr/patrick.flandrin/emd.html

    Unlike the numpypad method - this approach pads both the maxima and minima
    of the signal together.

    Parameters
    ----------
    indmin : ndarray
        location of minima in time
    indmax : ndarray
        location of maxima in time
    X : ndarray
        original time-series
    pad_width : int
        number of extra extrema to pad

    Returns
    -------
    tmin
        location of all minima (including padded and original points) in time
    xmin
        magnitude of each minima (including padded and original points)
    tmax
        location of all maxima (including padded and original points) in time
    xmax
        magnitude of each maxima (including padded and original points)

    """
    logger.debug("Padding {0} extrema in signal X {1} using method '{2}'".format(pad_width, X.shape, "rilling"))

    t = np.arange(len(X))

    # Pad START
    if indmax[0] < indmin[0]:
        # First maxima is before first minima
        if X[0] > X[indmin[0]]:
            # First value is larger than first minima - reflect about first MAXIMA
            logger.debug("L: max earlier than min, first val larger than first min")
            lmax = np.flipud(indmax[1 : pad_width + 1])
            lmin = np.flipud(indmin[:pad_width])
            lsym = indmax[0]
        else:
            # First value is smaller than first minima - reflect about first MINIMA
            logger.debug("L: max earlier than min, first val smaller than first min")
            lmax = np.flipud(indmax[:pad_width])
            lmin = np.r_[np.flipud(indmin[: pad_width - 1]), 0]
            lsym = 0

    else:
        # First minima is before first maxima
        if X[0] > X[indmax[0]]:
            # First value is larger than first minima - reflect about first MINIMA
            logger.debug("L: max later than min, first val larger than first max")
            lmax = np.flipud(indmax[:pad_width])
            lmin = np.flipud(indmin[1 : pad_width + 1])
            lsym = indmin[0]
        else:
            # First value is smaller than first minima - reflect about first MAXIMA
            logger.debug("L: max later than min, first val smaller than first max")
            lmin = np.flipud(indmin[:pad_width])
            lmax = np.r_[np.flipud(indmax[: pad_width - 1]), 0]
            lsym = 0

    # Pad STOP
    if indmax[-1] < indmin[-1]:
        # Last maxima is before last minima
        if X[-1] < X[indmax[-1]]:
            # Last value is larger than last minima - reflect about first MAXIMA
            logger.debug("R: max earlier than min, last val smaller than last max")
            rmax = np.flipud(indmax[-pad_width:])
            rmin = np.flipud(indmin[-pad_width - 1 : -1])
            rsym = indmin[-1]
        else:
            # First value is smaller than first minima - reflect about first MINIMA
            logger.debug("R: max earlier than min, last val larger than last max")
            rmax = np.r_[X.shape[0] - 1, np.flipud(indmax[-(pad_width - 2) :])]
            rmin = np.flipud(indmin[-(pad_width - 1) :])
            rsym = X.shape[0] - 1

    else:
        if X[-1] > X[indmin[-1]]:
            # Last value is larger than last minima - reflect about first MAXIMA
            logger.debug("R: max later than min, last val larger than last min")
            rmax = np.flipud(indmax[-pad_width - 1 : -1])
            rmin = np.flipud(indmin[-pad_width:])
            rsym = indmax[-1]
        else:
            # First value is smaller than first minima - reflect about first MINIMA
            logger.debug("R: max later than min, last val smaller than last min")
            rmax = np.flipud(indmax[-(pad_width - 1) :])
            rmin = np.r_[X.shape[0] - 1, np.flipud(indmin[-(pad_width - 2) :])]
            rsym = X.shape[0] - 1

    # Extrema values are ordered from largest to smallest,
    # lmin and lmax are the samples of the first {pad_width} extrema
    # rmin and rmax are the samples of the final {pad_width} extrema

    # Compute padded samples
    tlmin = 2 * lsym - lmin
    tlmax = 2 * lsym - lmax
    trmin = 2 * rsym - rmin
    trmax = 2 * rsym - rmax

    # tlmin and tlmax are the samples of the left/first padded extrema, in ascending order
    # trmin and trmax are the samples of the right/final padded extrema, in ascending order

    # Flip again if needed - don't really get what this is doing, will trust the source...
    if (tlmin[0] >= t[0]) or (tlmax[0] >= t[0]):
        msg = "Flipping start again - first min: {0}, first max: {1}, t[0]: {2}"
        logger.debug(msg.format(tlmin[0], tlmax[0], t[0]))
        if lsym == indmax[0]:
            lmax = np.flipud(indmax[:pad_width])
        else:
            lmin = np.flipud(indmin[:pad_width])
        lsym = 0
        tlmin = 2 * lsym - lmin
        tlmax = 2 * lsym - lmax

        if tlmin[0] >= t[0]:
            raise ValueError("Left min not padded enough. {0} {1}".format(tlmin[0], t[0]))
        if tlmax[0] >= t[0]:
            raise ValueError("Left max not padded enough. {0} {1}".format(trmax[0], t[0]))

    if (trmin[-1] <= t[-1]) or (trmax[-1] <= t[-1]):
        msg = "Flipping end again - last min: {0}, last max: {1}, t[-1]: {2}"
        logger.debug(msg.format(trmin[-1], trmax[-1], t[-1]))
        if rsym == indmax[-1]:
            rmax = np.flipud(indmax[-pad_width - 1 : -1])
        else:
            rmin = np.flipud(indmin[-pad_width - 1 : -1])
        rsym = len(X)
        trmin = 2 * rsym - rmin
        trmax = 2 * rsym - rmax

        if trmin[-1] <= t[-1]:
            raise ValueError("Right min not padded enough. {0} {1}".format(trmin[-1], t[-1]))
        if trmax[-1] <= t[-1]:
            raise ValueError("Right max not padded enough. {0} {1}".format(trmax[-1], t[-1]))

    # Stack and return padded values
    ret_tmin = np.r_[tlmin, t[indmin], trmin]
    ret_tmax = np.r_[tlmax, t[indmax], trmax]

    ret_xmin = np.r_[X[lmin], X[indmin], X[rmin]]
    ret_xmax = np.r_[X[lmax], X[indmax], X[rmax]]

    # Quick check that interpolation won't explode
    if np.all(np.diff(ret_tmin) > 0) is False:
        logger.warning("Minima locations not strictly ascending - interpolation will break")
        raise ValueError("Extrema locations not strictly ascending!!")
    if np.all(np.diff(ret_tmax) > 0) is False:
        logger.warning("Maxima locations not strictly ascending - interpolation will break")
        raise ValueError("Extrema locations not strictly ascending!!")

    return ret_tmin, ret_xmin, ret_tmax, ret_xmax


def get_padded_extrema(
    x, pad_width=2, mode="peaks", parabolic_extrema=False, loc_pad_opts=None, mag_pad_opts=None, method="rilling"
):
    """Identify and pad the extrema in a signal.

    This function returns a set of extrema from a signal including padded
    extrema at the edges of the signal. Padding is carried out using numpy.pad.

    Parameters
    ----------
    x : ndarray
        Input signal
    pad_width : int >= 0
        Number of additional extrema to add to the start and end
    mode : {'peaks', 'troughs', 'abs_peaks', 'both'}
        Switch between detecting peaks, troughs, peaks in the abs signal or
        both peaks and troughs
    method : {'rilling', 'numpypad'}
        Which padding method to use
    parabolic_extrema : bool
        Flag indicating whether extrema positions should be refined by parabolic interpolation
    loc_pad_opts : dict
        Optional dictionary of options to be passed to np.pad when padding extrema locations
    mag_pad_opts : dict
        Optional dictionary of options to be passed to np.pad when padding extrema magnitudes

    Returns
    -------
    locs : ndarray
        location of extrema in samples
    mags : ndarray
        Magnitude of each extrema

    See Also
    --------
    emd.sift.interp_envelope
    emd.sift._pad_extrema_numpy
    emd.sift._pad_extrema_rilling

    Notes
    -----
    The 'abs_peaks' mode is not compatible with the 'rilling' method as rilling
    must identify all peaks and troughs together.

    """
    if (mode == "abs_peaks") and (method == "rilling"):
        msg = "get_padded_extrema mode 'abs_peaks' is incompatible with method 'rilling'"
        raise ValueError(msg)

    if x.ndim == 2:
        x = x[:, 0]

    if mode == "both" or method == "rilling":
        max_locs, max_ext = _find_extrema(x, parabolic_extrema=parabolic_extrema)
        min_locs, min_ext = _find_extrema(-x, parabolic_extrema=parabolic_extrema)
        min_ext = -min_ext
        logger.debug("found {0} minima and {1} maxima on mode {2}".format(len(min_locs), len(max_locs), mode))
    elif mode == "peaks":
        max_locs, max_ext = _find_extrema(x, parabolic_extrema=parabolic_extrema)
        logger.debug("found {0} maxima on mode {1}".format(len(max_locs), mode))
    elif mode == "troughs":
        max_locs, max_ext = _find_extrema(-x, parabolic_extrema=parabolic_extrema)
        max_ext = -max_ext
        logger.debug("found {0} minima on mode {1}".format(len(max_locs), mode))
    elif mode == "abs_peaks":
        max_locs, max_ext = _find_extrema(np.abs(x), parabolic_extrema=parabolic_extrema)
        logger.debug("found {0} extrema on mode {1}".format(len(max_locs), mode))
    else:
        raise ValueError("Mode {0} not recognised by get_padded_extrema".format(mode))

    # Return nothing if we don't have enough extrema
    if (len(max_locs) == 0) or (max_locs.size <= 1):
        logger.debug("Not enough extrema to pad.")
        return None, None
    elif (mode == "both" or method == "rilling") and len(min_locs) <= 1:
        logger.debug("Not enough extrema to pad 2.")
        return None, None

    # Run the padding by requested method
    if pad_width == 0:
        if mode == "both":
            ret = (min_locs, min_ext, max_locs, max_ext)
        elif mode == "troughs" and method == "rilling":
            ret = (min_locs, min_ext)
        else:
            ret = (max_locs, max_ext)
    elif method == "numpypad":
        ret = _pad_extrema_numpy(max_locs, max_ext, x.shape[0], pad_width, loc_pad_opts, mag_pad_opts)
        if mode == "both":
            ret2 = _pad_extrema_numpy(min_locs, min_ext, x.shape[0], pad_width, loc_pad_opts, mag_pad_opts)
            ret = (ret2[0], ret2[1], ret[0], ret[1])
    elif method == "rilling":
        ret = _pad_extrema_rilling(min_locs, max_locs, x, pad_width)
        # Inefficient to use rilling for just peaks or troughs, but handle it
        # just in case.
        if mode == "peaks":
            ret = ret[2:]
        elif mode == "troughs":
            ret = ret[:2]

    return ret


def _run_scipy_interp(locs, pks, lenx, interp_method="splrep", trim=True):
    from scipy import interpolate as interp

    # Run interpolation on envelope
    t = np.arange(locs[0], locs[-1])
    if interp_method == "splrep":
        f = interp.splrep(locs, pks)
        env = interp.splev(t, f)
    elif interp_method == "mono_pchip":
        pchip = interp.PchipInterpolator(locs, pks)
        env = pchip(t)
    elif interp_method == "pchip":
        pchip = interp.pchip(locs, pks)
        env = pchip(t)

    if trim:
        t_max = np.arange(locs[0], locs[-1])
        tinds = np.logical_and((t_max >= 0), (t_max < lenx))
        env = np.array(env[tinds])

        if env.shape[0] != lenx:
            msg = "Envelope length does not match input data {0} {1}"
            raise ValueError(msg.format(env.shape[0], lenx))

    return env


def interp_envelope(x, mode="both", interp_method="splrep", extrema_opts=None, ret_extrema=False, trim=True):
    """Interpolate the amplitude envelope of a signal.

    Parameters
    ----------
    x : ndarray
        Input signal
    mode : {'upper','lower','combined'}
         Flag to set which envelope should be computed (Default value = 'upper')
    interp_method : {'splrep','pchip','mono_pchip'}
         Flag to indicate which interpolation method should be used (Default value = 'splrep')

    Returns
    -------
    ndarray
        Interpolated amplitude envelope

    """
    if not extrema_opts:  # Empty dict evaluates to False
        extrema_opts = {"pad_width": 2, "loc_pad_opts": None, "mag_pad_opts": None}
    else:
        extrema_opts = extrema_opts.copy()  # Don't work in place...

    logger.debug("Interpolating '{0}' with method '{1}'".format(mode, interp_method))

    if interp_method not in ["splrep", "mono_pchip", "pchip"]:
        raise ValueError("Invalid interp_method value")

    if mode == "upper":
        extr = get_padded_extrema(x, mode="peaks", **extrema_opts)
    elif mode == "lower":
        extr = get_padded_extrema(x, mode="troughs", **extrema_opts)
    elif (mode == "both") or (extrema_opts.get("method", "") == "rilling"):
        extr = get_padded_extrema(x, mode="both", **extrema_opts)
    elif mode == "combined":
        extr = get_padded_extrema(x, mode="abs_peaks", **extrema_opts)
    else:
        raise ValueError("Mode not recognised. Use mode= 'upper'|'lower'|'combined'")

    if extr[0] is None:
        if mode == "both":
            return None, None
        else:
            return None

    if mode == "both":
        lower = _run_scipy_interp(extr[0], extr[1], lenx=x.shape[0], trim=trim, interp_method=interp_method)
        upper = _run_scipy_interp(extr[2], extr[3], lenx=x.shape[0], trim=trim, interp_method=interp_method)
        env = (upper, lower)
    else:
        env = _run_scipy_interp(extr[0], extr[1], lenx=x.shape[0], interp_method=interp_method, trim=trim)

    if ret_extrema:
        return env, extr
    else:
        return env


def sd_stop(proto_imf, prev_imf, sd=0.2, niters=None):
    """Compute the sd sift stopping metric.

    Parameters
    ----------
    proto_imf : ndarray
        A signal which may be an IMF
    prev_imf : ndarray
        The previously identified IMF
    sd : float
        The stopping threshold
    niters : int
        Number of sift iterations currently completed
    niters : int
        Number of sift iterations currently completed

    Returns
    -------
    bool
        A flag indicating whether to stop siftingg
    float
        The SD metric value

    """
    metric = np.sum((proto_imf - prev_imf) ** 2) / np.sum(proto_imf**2)

    stop = metric < sd

    if stop:
        logger.verbose("Sift stopped by SD-thresh in {0} iters with sd {1}".format(niters, metric))
    else:
        logger.debug("SD-thresh stop metric evaluated at iter {0} is : {1}".format(niters, metric))

    return stop, metric


def rilling_stop(upper_env, lower_env, sd1=0.05, sd2=0.5, tol=0.05, niters=None):
    """Compute the Rilling et al 2003 sift stopping metric.

    This metric tries to guarantee globally small fluctuations in the IMF mean
    while taking into account locally large excursions that may occur in noisy
    signals.

    Parameters
    ----------
    upper_env : ndarray
        The upper envelope of a proto-IMF
    lower_env : ndarray
        The lower envelope of a proto-IMF
    sd1 : float
        The maximum threshold for globally small differences from zero-mean
    sd2 : float
        The maximum threshold for locally large differences from zero-mean
    tol : float (0 < tol < 1)
        (1-tol) defines the proportion of time which may contain large deviations
        from zero-mean
    niters : int
        Number of sift iterations currently completed

    Returns
    -------
    bool
        A flag indicating whether to stop siftingg
    float
        The SD metric value

    Notes
    -----
    This method is described in section 3.2 of:
    Rilling, G., Flandrin, P., & Goncalves, P. (2003, June). On empirical mode
    decomposition and its algorithms. In IEEE-EURASIP workshop on nonlinear
    signal and image processing (Vol. 3, No. 3, pp. 8-11). NSIP-03, Grado (I).
    http://perso.ens-lyon.fr/patrick.flandrin/NSIP03.pdf

    """
    avg_env = (upper_env + lower_env) / 2
    amp = np.abs(upper_env - lower_env) / 2

    eval_metric = np.abs(avg_env) / amp

    metric = np.mean(eval_metric > sd1)
    continue1 = metric > tol
    continue2 = np.any(eval_metric > sd2)

    stop = (continue1 or continue2) == False  # noqa: E712

    if stop:
        logger.verbose("Sift stopped by Rilling-metric in {0} iters (val={1})".format(niters, metric))
    else:
        logger.debug("Rilling stop metric evaluated at iter {0} is : {1}".format(niters, metric))

    return stop, metric


def _energy_difference(imf, residue):
    """Compute energy change in IMF during a sift.

    Parameters
    ----------
    imf : ndarray
        IMF to be evaluated
    residue : ndarray
        Remaining signal after IMF removal

    Returns
    -------
    float
        Energy difference in decibels

    Notes
    -----
    This function is used during emd.sift.energy_stop to implement the
    energy-difference sift-stopping method defined in section 3.2.4 of
    https://doi.org/10.1016/j.ymssp.2007.11.028

    """
    sumsqr = np.sum(imf**2)
    imf_energy = 20 * np.log10(sumsqr, where=sumsqr > 0)
    sumsqr = np.sum(residue**2)
    resid_energy = 20 * np.log10(sumsqr, where=sumsqr > 0)
    return imf_energy - resid_energy


def energy_stop(imf, residue, thresh=50, niters=None):
    """Compute energy change in IMF during a sift.

    The energy in the IMFs are compared to the energy at the start of sifting.
    The sift terminates once this ratio reaches a predefined threshold.

    Parameters
    ----------
    imf : ndarray
        IMF to be evaluated
    residue : ndarray
        Average of the upper and lower envelopes
    thresh : float
        Energy ratio threshold (default=50)
    niters : int
        Number of sift iterations currently completed

    Returns
    -------
    bool
        A flag indicating whether to stop siftingg
    float
        Energy difference in decibels

    Notes
    -----
    This function implements the energy-difference sift-stopping method defined
    in section 3.2.4 of https://doi.org/10.1016/j.ymssp.2007.11.028

    """
    diff = _energy_difference(imf, residue)
    stop = bool(diff > thresh)

    if stop:
        logger.debug("Sift stopped by Energy Ratio in {0} iters with difference of {1}dB".format(niters, diff))
    else:
        logger.debug("Energy Ratio evaluated at iter {0} is : {1}dB".format(niters, diff))

    return stop, diff


def fixed_stop(niters, max_iters):
    """Compute the fixed-iteraiton sift stopping metric.

    Parameters
    ----------
    niters : int
        Number of sift iterations currently completed
    max_iters : int
        Maximum number of sift iterations to be completed

    Returns
    -------
    bool
        A flag indicating whether to stop siftingg

    """
    stop = bool(niters == max_iters)

    if stop:
        logger.debug("Sift stopped at fixed number of {0} iterations".format(niters))

    return stop


def get_next_imf(
    x,
    env_step_size=1,
    max_iters=1000,
    energy_thresh=50,
    stop_method="sd",
    sd_thresh=0.1,
    rilling_thresh=(0.05, 0.5, 0.05),
    envelope_opts=None,
    extrema_opts=None,
):
    """Compute the next IMF from a data set.

    This is a helper function used within the more general sifting functions.

    Parameters
    ----------
    x : ndarray [nsamples x 1]
        1D input array containing the time-series data to be decomposed
    env_step_size : float
        Scaling of envelope prior to removal at each iteration of sift. The
        average of the upper and lower envelope is muliplied by this value
        before being subtracted from the data. Values should be between
        0 > x >= 1 (Default value = 1)
    max_iters : int > 0
        Maximum number of iterations to compute before throwing an error
    energy_thresh : float > 0
        Threshold for energy difference (in decibels) between IMF and residual
        to suggest stopping overall sift. (Default is None, recommended value is 50)
    stop_method : {'sd','rilling','fixed'}
        Flag indicating which metric to use to stop sifting and return an IMF.
    sd_thresh : float
        Used if 'stop_method' is 'sd'. The threshold at which the sift of each
        IMF will be stopped. (Default value = .1)
    rilling_thresh : tuple
        Used if 'stop_method' is 'rilling', needs to contain three values (sd1, sd2, alpha).
        An evaluation function (E) is defined by dividing the residual by the
        mode amplitude. The sift continues until E < sd1 for the fraction
        (1-alpha) of the data, and E < sd2 for the remainder.
        See section 3.2 of http://perso.ens-lyon.fr/patrick.flandrin/NSIP03.pdf

    Returns
    -------
    proto_imf : ndarray
        1D vector containing the next IMF extracted from x
    continue_flag : bool
        Boolean indicating whether the sift can be continued beyond this IMF

    Other Parameters
    ----------------
    envelope_opts : dict
        Optional dictionary of keyword arguments to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema

    See Also
    --------
    emd.sift.sift
    emd.sift.interp_envelope

    """
    x = ensure_1d_with_singleton([x], ["x"], "get_next_imf")

    if envelope_opts is None:
        envelope_opts = {}

    proto_imf = x.copy()

    continue_imf = True
    continue_flag = True
    niters = 0
    while continue_imf:
        niters += 1

        upper, lower = interp_envelope(proto_imf, mode="both", **envelope_opts, extrema_opts=extrema_opts)

        # If upper or lower are None we should stop sifting altogether
        if upper is None or lower is None:
            continue_flag = False
            continue_imf = False
            logger.debug("Finishing sift: IMF has no extrema")
            continue

        # Find local mean
        avg = np.mean([upper, lower], axis=0)[:, None]

        # Remove local mean estimate from proto imf
        x1 = proto_imf - avg

        # Stop sifting if we pass threshold
        if stop_method == "sd":
            # Cauchy criterion
            stop, _ = sd_stop(proto_imf, x1, sd=sd_thresh, niters=niters)
        elif stop_method == "rilling":
            # Rilling et al 2003
            stop, _ = rilling_stop(
                upper, lower, niters=niters, sd1=rilling_thresh[0], sd2=rilling_thresh[1], tol=rilling_thresh[2]
            )
        elif stop_method == "energy":
            # Rato et al 2008
            # Compare energy of signal at start of sift with energy of envelope average
            stop, _ = energy_stop(x, avg, thresh=energy_thresh, niters=niters)
        elif stop_method == "fixed":
            stop = fixed_stop(niters, max_iters)
        else:
            raise ValueError("stop_method '{0}' not recogised".format(stop_method))

        if stop:
            proto_imf = x1.copy()
            continue_imf = False
            continue

        proto_imf = proto_imf - (env_step_size * avg)

    if proto_imf.ndim == 1:
        proto_imf = proto_imf[:, None]

    return proto_imf, continue_flag


def sift(x, sift_thresh=1e-8, max_imfs=None, verbose=None, imf_opts=None, envelope_opts=None, extrema_opts=None):
    """Compute Intrinsic Mode Functions from an input data vector.

    This function implements the original sift algorithm [1]_.

    Parameters
    ----------
    x : ndarray
        1D input array containing the time-series data to be decomposed
    sift_thresh : float
         The threshold at which the overall sifting process will stop. (Default value = 1e-8)
    max_imfs : int
         The maximum number of IMFs to compute. (Default value = None)

    Returns
    -------
    imf: ndarray
        2D array [samples x nimfs] containing he Intrisic Mode Functions from the decomposition of x.

    Other Parameters
    ----------------
    imf_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_next_imf
    envelope_opts : dict
        Optional dictionary of keyword options to be passed to emd.interp_envelope
    extrema_opts : dict
        Optional dictionary of keyword options to be passed to emd.get_padded_extrema
    verbose : {None,'CRITICAL','WARNING','INFO','DEBUG'}
        Option to override the EMD logger level for a call to this function.

    See Also
    --------
    emd.sift.get_next_imf
    emd.sift.get_config

    Notes
    -----
    The classic sift is computed by passing an input vector with all options
    left to default

    >>> imf = emd.sift.sift(x)

    The sift can be customised by passing additional options, here we only
    compute the first four IMFs.

    >>> imf = emd.sift.sift(x, max_imfs=4)

    More detailed options are passed as dictionaries which are passed to the
    relevant lower-level functions. For instance `imf_opts` are passed to
    `get_next_imf`.

    >>> imf_opts = {'env_step_size': 1/3, 'stop_method': 'rilling'}
    >>> imf = emd.sift.sift(x, max_imfs=4, imf_opts=imf_opts)

    A modified dictionary of all options can be created using `get_config`.
    This can be modified and used by unpacking the options into a `sift` call.

    >>> conf = emd.sift.get_config('sift')
    >>> conf['max_imfs'] = 4
    >>> conf['imf_opts'] = imf_opts
    >>> imfs = emd.sift.sift(x, **conf)

    References
    ----------
    .. [1] Huang, N. E., Shen, Z., Long, S. R., Wu, M. C., Shih, H. H., Zheng,
       Q., … Liu, H. H. (1998). The empirical mode decomposition and the Hilbert
       spectrum for nonlinear and non-stationary time series analysis. Proceedings
       of the Royal Society of London. Series A: Mathematical, Physical and
       Engineering Sciences, 454(1971), 903–995.
       https://doi.org/10.1098/rspa.1998.0193

    """
    if not imf_opts:
        imf_opts = {"env_step_size": 1, "sd_thresh": 0.1}

    x = ensure_1d_with_singleton([x], ["x"], "sift")

    continue_sift = True
    layer = 0

    proto_imf = x.copy()

    while continue_sift:
        logger.info("sifting IMF : {0}".format(layer))

        next_imf, continue_sift = get_next_imf(
            proto_imf, envelope_opts=envelope_opts, extrema_opts=extrema_opts, **imf_opts
        )

        if layer == 0:
            imf = next_imf
        else:
            imf = np.concatenate((imf, next_imf), axis=1)

        proto_imf = x - imf.sum(axis=1)[:, None]
        layer += 1

        if max_imfs is not None and layer == max_imfs:
            logger.info("Finishing sift: reached max number of imfs ({0})".format(layer))
            continue_sift = False

        if np.abs(next_imf).sum() < sift_thresh:
            logger.info("Finishing sift: reached threshold {0}".format(np.abs(next_imf).sum()))
            continue_sift = False

    return imf


def hilbert_huang_transform(
    signal: pd.Series,
    sift_thresh: float = 1e-2,
    max_num_imfs: Optional[int] = None,
    error_tolerance: float = 0.5,
    return_trend: bool = True,
) -> pd.Series:
    r"""
    Perform the Hilbert-Huang Transform (HHT) to find the trend of a signal.

    The Hilbert-Huang Transform is a technique that combines Empirical Mode Decomposition (EMD)
    and the Hilbert Transform to analyze non-stationary and non-linear time series data,
    where the Hilbert transform is applied to each mode after having performed the EMD.
    Non-stationary signals are signals that vary in frequency and amplitude over time,
    and cannot be adequately represented by fixed parameters, whereas non-linear signals are signals
    that cannot be represented by a linear function and can exhibit complex and unpredictable behavior.
    Signals from different physical phenomena in the world are rarely purely linear or
    stationary, and so the HHT is a useful tool for analyzing real-world signals.

    Given their complexity, it is often difficult to study the entire signals as a whole.
    Therefore, the HHT aims to capture the time-varying nature and non-linear dynamics of such signals by
    decomposing them into individual oscillatory components by the use the EMD (and the Hilbert transform).
    These components are oscillatory modes of the full signal with well-defined instantaneous frequencies called
    Intrinsic Mode Functions (IMFs).

    \textbf{Empirical Mode Decomposition} is a technique used to decompose a given signal into a set of
    intrinsic mode functions. To perform EMD, we start by finding the local maxima and minima in the signal.
    Then, we smoothly connect these points to create upper and lower envelopes using spline interpolation
    that capture the signal's overall trend. Next, we calculate the average of these envelopes and subtract
    it from the original signal, resulting in the first IMF—a high-frequency component. We repeat this process,
    sifting out the IMFs one by one until each IMF satisfies the wiggling and envelope criteria. The final result
    is a collection of IMFs, ordered by their frequencies, that represent the different oscillatory modes present
    in the original signal.

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
    if sift_thresh <= 0:
        raise ValueError("The sifting threshold must be a number higher than zero")
    if max_num_imfs is not None and max_num_imfs <= 0:
        raise ValueError("The maximum number of oscillatory components must be an integer higher than zero")
    if error_tolerance <= 0:
        raise ValueError("The energy tolerance must be higher than zero")

    signal_array = signal.to_numpy()

    emd_ = EMD()
    # imfs = emd_(signal_array)
    # imfs = sift(signal_array, sift_threshold=sift_thresh, imf_limit=max_num_imfs)
    imfs = sift(signal_array, sift_thresh=sift_thresh, max_imfs=max_num_imfs)

    imfs_old = emd.sift.sift(signal_array, sift_thresh, max_num_imfs)
    # Convert to numpy array if it's not already
    if not isinstance(imfs_old, np.ndarray):
        imfs_old = np.array(imfs_old)

    if max_num_imfs is not None and max_num_imfs <= len(imfs):
        imfs = imfs[:max_num_imfs]

    analytic_signals = []
    for imf in imfs:
        analytic_signal = hilbert(imf)
        analytic_signals.append(analytic_signal)

    print("signal_array: ", signal_array.shape)
    # print("analytic_signals: ", analytic_signals)
    # for analytic_signal in analytic_signals:
    #     assert len(analytic_signal) == len(signal_array), "Mismatch in length for an analytic signal"

    total_number_of_imf: int = imfs.shape[1]
    # total_number_of_imf_old: int = imfs_old.shape[1]

    index_of_the_last_imf: int = total_number_of_imf - 1
    # index_of_the_last_imf_old: int = total_number_of_imf_old - 1
    # significant_imf_index: int = index_of_the_last_imf

    significant_imf_index_rho = len(imfs) - 1
    significant_imf_index_rho_old = len(imfs_old) - 1

    phase = np.unwrap(np.angle(analytic_signals))
    print("phase WWWWWWWWW: ", len(phase))
    # Calculate the timestep based on the actual time range
    dt = np.mean(np.diff(get_timestamps(signal, "s").to_numpy()))
    # dt = (signal_array[-1] - signal_array[0]) / len(
    #     signal_array
    # )
    frequency = np.gradient(phase) / (2 * np.pi * dt)
    amplitude = np.abs(analytic_signals)
    print("amplitude WWWWWWWWWWWW: ", len(amplitude))
    timestamps = get_timestamps(signal, "s")
    sample_rate_hz = int(1 / (np.mean(np.diff(timestamps))))

    # compute phase, frequency and amplitude
    # phase, frequency, amplitude = compute_frequency_attributes(imf, sample_rate=sample_rate_hz, calculation_method="hilbert")

    # compute Hilbert-Huang spectrum for each IMF separately and sum over time to get Hilbert marginal spectrum
    # hht = np.sum(amplitude * np.exp(frequency * np.arange(len(signal_array)) * 1j), axis=1)
    frequency_array = np.array(frequency).T  # transpose to get the right shape
    avg_frequency_array = np.mean(frequency_array, axis=-1)
    amplitude_array = np.array(amplitude)  # transpose to get the right shape

    flat_amplitude = amplitude_array.flatten()
    # flat_avg_frequency = np.tile(avg_frequency_array.T, amplitude_array.T).flatten()
    flat_avg_frequency = avg_frequency_array.flatten()
    timestamps = get_timestamps(signal, "s")

    print("length of signal array: ", timestamps.shape)
    print("Shape of amplitude_array:", flat_amplitude.shape)
    print("Shape of frequency_array:", flat_avg_frequency.shape)

    # the hilbert huang spectrum
    # hht = abs(np.sum(flat_amplitude * np.exp(flat_avg_frequency * np.arange(len(signal_array)) * 1j), axis=1))
    # hht = abs(np.sum(flat_amplitude * np.exp(flat_avg_frequency * np.arange(len(signal_array)) * 1j)), axis=1)

    _, hht = hilbert_huang_spectrum(
        flat_avg_frequency, flat_amplitude, aggregate_imfs=False, aggregate_time=False, data_sample_rate=sample_rate_hz
    )

    print("Shape of final_spectra:", hht.shape)

    assert flat_amplitude.shape == flat_avg_frequency.shape, "Mismatch between frequency and amplitude arrays"
    # assert flat_amplitude.shape == hht.shape, "Mismatch between frequency and amplitude arrays"

    # timestamps = get_timestamps(signal, "s")
    sample_rate_hz = int(1 / (np.mean(np.diff(timestamps))))

    # phase_old, frequency_old, amplitude_old = emd.spectra.frequency_transform(
    #     imfs_old, sample_rate=sample_rate_hz, method="hilbert"
    # )

    # Convert to numpy array if it's not already
    # if not isinstance(frequency_old, np.ndarray):
    #     frequency_old = np.array(frequency_old)
    # if not isinstance(amplitude_old, np.ndarray):
    #     amplitude_old = np.array(amplitude_old)

    # flat_frequency_old = frequency_old.flatten()
    # flat_amplitude_old = amplitude_old.flatten()

    # _, hht_old = emd.spectra.hilberthuang(
    #     flat_avg_frequency, flat_amplitude, sum_imfs=False, sum_time=False, sample_rate=sample_rate_hz
    # )

    # print("Shape of frequency_old:", frequency_old.shape)
    # print("Shape of amplitude_old:", amplitude_old.shape)
    # _, hht_old = emd.spectra.hilberthuang(
    #     frequency_old, amplitude_old, sum_imfs=False, sum_time=False, sample_rate=sample_rate_hz
    # )

    # print("Shape of hht_old:", hht_old.shape)

    # significant_imf_index = len(imfs) - 1

    # for i in range(index_of_the_last_imf - 1, -1, -1):
    #     rho = np.sum(final_spectra[:, i] * final_spectra[:, i + 1]) / (
    #         np.sum(final_spectra[:, i]) * np.sum(final_spectra[:, i + 1])
    #     )
    #     rho_old = np.sum(hht_old[:, i] * hht_old[:, i + 1]) / (np.sum(hht_old[:, i]) * np.sum(hht_old[:, i + 1]))

    #     if rho < error_tolerance:
    #         break
    #     else:
    #         significant_imf_index_rho -= 1

    #     if rho_old < error_tolerance:
    #         break
    #     else:
    #         significant_imf_index_rho_old -= 1

    # Loop for rho calculations
    for i in range(index_of_the_last_imf - 1, -1, -1):
        rho = np.sum(hht[:, i] * hht[:, i + 1]) / (np.sum(hht[:, i]) * np.sum(hht[:, i + 1]))
        print("rho: ", rho)

        if rho < error_tolerance:
            break
        else:
            significant_imf_index_rho -= 1
            print("significant_imf_index_rho: ", significant_imf_index_rho)

    # Loop for rho_old calculations
    # for i in range(index_of_the_last_imf_old - 1, -1, -1):
    #     rho_old = np.sum(hht_old[:, i] * hht_old[:, i + 1]) / (np.sum(hht_old[:, i]) * np.sum(hht_old[:, i + 1]))

    #     if rho_old < error_tolerance:
    #         break
    #     else:
    #         significant_imf_index_rho_old -= 1
    #         # print("significant_imf_index_rho_old: ", significant_imf_index_rho_old)

    # Compute trends for both rho and rho_old
    print(imfs[significant_imf_index_rho:])
    trend_rho = np.sum(imfs[significant_imf_index_rho:], axis=0, dtype=np.float64)
    print("imfs[significant_imf_index_rho:]: ", imfs[significant_imf_index_rho:])
    print("trend_rho: ", trend_rho)
    print("signal.index: ", signal.index)
    trend_series_rho = pd.Series(trend_rho, index=signal.index)
    print("trend_series_rho: ", trend_series_rho)

    trend_rho_old = np.sum(imfs_old[significant_imf_index_rho_old:], axis=0)
    print("imfs_old[significant_imf_index_rho_old:]: ", imfs_old[significant_imf_index_rho_old:])
    print("trend_rho_old: ", trend_rho_old)
    print("signal.index: ", signal.index)
    # trend_series_rho_old = pd.Series(trend_rho_old)  # , index=signal.index)

    # Modify the return to return both trends or their detrended signals based on the return_trend flag
    result_rho = trend_series_rho if return_trend else signal - trend_series_rho
    # result_rho_old = trend_series_rho_old if return_trend else signal - trend_series_rho_old

    return result_rho
