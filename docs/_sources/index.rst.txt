.. _gallery:

Cognite's Industrial Data Science Library
=========================================

This is Cognite's collection of data science algorithms and models. Its objective is twofold. First, empower domain
experts to conduct exploratory work, root cause analysis, and analyze data without the requirement to code by driving
the industrial data science and analytics engine behind |charts_link|. Second, curate a collection industrial relevant
data science algorithms to be used by any data scientist. For more information, check |charts_docs|.

.. |charts_link| raw:: html

   <a href="https://charts.cogniteapp.com/" target="_blank">Cognite Charts</a>

.. |charts_docs| raw:: html

   <a href="https://docs.cognite.com/cdf/charts.html" target="_blank">Charts documentation page</a>

Installation
------------

To install the core part of the library, where dependencies are `numpy`, `scipy`` and `pandas`, run:

.. code-block:: bash

   pip install indsl

InDSL also includes extra functionality that that requires additonal dependencies. The extra functionality includes:

- Numba, for pre-compiled functions and faster execution
- Plot, for plotting functionality that depends on `matplotlib`
- Modeling, for functionality that depends on `csaps` and `kneed`.
- Stats, for statistical functionality that depends on `statsmodels`
- Scikit, for machine learning functionality that depends on `scikit-image` and `scikit-learn`
- Fluids, for fluid dynamics calculations and simulations.

To install extra functionality, run:

.. code-block:: bash

   pip install indsl[numba,plot,modeling,stats,scikit,fluids]

To install all the extra functionality, run:

.. code-block:: bash

   pip install indsl[all]

.. toctree::
   :maxdepth: 2
   :caption: Toolboxes

   data_quality
   detect
   drilling
   equipment
   filter
   fluid_dynamics
   forecast
   numerical_calculus
   oil_and_gas
   resample
   signals
   smooth
   statistics
   sustainability
   ts_utils

.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/data_quality/index
   auto_examples/detect/index
   auto_examples/equipment/index
   auto_examples/filter/index
   auto_examples/fluid_dynamics/index
   auto_examples/forecast/index
   auto_examples/numerical_calculus/index
   auto_examples/oil_and_gas/index
   auto_examples/resample/index
   auto_examples/signals/index
   auto_examples/smooth/index
   auto_examples/statistics/index
   auto_examples/sustainability/index
   auto_examples/versioning/index

.. toctree::
   :maxdepth: 2
   :caption: Developers

   contribute
   code_of_conduct
   dev_tools
   CHANGELOG

Indices and tables
==================

* :ref:`genindex`

..
    * :ref:`modindex`
    * :ref:`search`
