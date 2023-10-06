Data Quality
############

Completeness
************

Completeness Score
==================
.. autofunction:: indsl.data_quality.completeness.completeness_score

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_completeness.py`

Data Gaps Detection
===================

Using Z scores
--------------
.. autofunction:: indsl.data_quality.gaps_identification_z_scores

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_gaps_identification.py`


Using modified Z scores
-----------------------
.. autofunction:: indsl.data_quality.gaps_identification_modified_z_scores

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_gaps_identification.py`

Using the interquartile range method
------------------------------------
.. autofunction:: indsl.data_quality.gaps_identification_iqr

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_gaps_identification.py`

Using a time delta threshold
----------------------------
.. autofunction:: indsl.data_quality.gaps_identification_threshold

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_gaps_identification.py`


Low data density
================

Using Z scores
--------------
.. autofunction:: indsl.data_quality.low_density_identification_z_scores

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_low_density_identification.py`

Using modified Z scores
-----------------------
.. autofunction:: indsl.data_quality.low_density_identification_modified_z_scores

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_low_density_identification.py`
   
Using the interquartile range method
------------------------------------
.. autofunction:: indsl.data_quality.low_density_identification_iqr

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_low_density_identification.py`

Using a density threshold
-------------------------
.. autofunction:: indsl.data_quality.low_density_identification_threshold

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_low_density_identification.py`


Rolling standard deviation of time delta
========================================
.. autofunction:: indsl.data_quality.rolling_stddev_timedelta

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_rolling_stddev_timedelta.py`

Validity
********

Extreme Outliers Removal
========================
.. autofunction:: indsl.data_quality.extreme

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_extreme_outlier.py`


Out of Range Values
===================
.. autofunction:: indsl.data_quality.outliers.out_of_range

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_out_of_range.py`


Value Decrease Indication
=========================
.. autofunction:: indsl.data_quality.value_decrease_check

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_value_decrease_check.py`

Datapoint difference over a period of time
==========================================
.. autofunction:: indsl.data_quality.datapoint_diff_over_time_period

.. topic:: Examples:

   * :ref:`sphx_glr_auto_examples_data_quality_plot_datapoint_diff.py`
