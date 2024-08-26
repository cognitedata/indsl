Signals
=======

Module for Generating Industrial Grade Synthetic Signals
--------------------------------------------------------

Industrial time series (i.e., sensor data from facilities) are commonly sampled at irregular intervals (non-uniform
time stamps), contain data gaps, noise of different characteristics, and many other data quality flaws. The objective of
this module is to offer multiple type of synthetic signals and methods to introduce data quality features similar to
those observed in real industrial time series.

Line time series
^^^^^^^^^^^^^^^^
.. autofunction:: indsl.signals.generator.line

Constant value time series
^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: indsl.signals.generator.const_value

Sine wave
^^^^^^^^^
.. autofunction:: indsl.signals.generator.sine_wave

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_signals_plot_wavy_signals.py`

Perturb the index of a time series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: indsl.signals.generator.perturb_timestamp

Create data gaps in a time series
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: indsl.signals.generator.insert_data_gaps

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_signals_plot_synthetic_gaps.py`


Noise Generators
-----------------

White noise
^^^^^^^^^^^
.. autofunction:: indsl.signals.noise.white_noise

.. topic:: Examples with noise generators:

   * :ref:`sphx_glr_auto_examples_signals_plot_wavy_signals.py`

Brownian noise
^^^^^^^^^^^^^^
.. autofunction:: indsl.signals.generator.wave_with_brownian_noise

.. topic:: Examples with noise generators:

   * :ref:`sphx_glr_auto_examples_signals_plot_wavy_signals.py`

Polynomial Generators
---------------------

Univariate Polynomial
^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: indsl.signals.polynomial.univariate_polynomial

.. topic:: Examples with univariate polynomial generators:

   * :ref:`sphx_glr_auto_examples_signals_plot_univariate_polynomial.py`   


Sequence interpolation
----------------------

1D interpolation of a sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: indsl.signals.sequence_interpolation.sequence_interpolation_1d

2D interpolation of a sequence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: indsl.signals.sequence_interpolation.sequence_interpolation_2d

