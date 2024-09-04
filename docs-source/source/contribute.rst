============
Contributing
============

This project is a community effort and contributions are welcomed. InDSL is publicly available and open for contributions 
`here <https://github.com/cognitedata/indsl>`_. Engage on our community site, `Cognite Hub <https://hub.cognite.com/>`_, 
for discussion, suggestions and questions about InDSL.

The main objective of the InDLS is to **provide industrial domain experts and data scientist with a rich library of
algorithms to speed up their work**. Therefore, we highly encourage data scientists with industrial domain knowledge
to contribute algorithms and models within their area of expertise. *We are industry and scientific domain
agnostic*. We accept any type of algorithm that improves the industrial data science experience and development.

Given the above, we are picky when it comes to adding new algorithms and how we document them. We want to speed up our
user's tasks with algorithms that minimize their exploratory and analytic work. We strive to include
methods that will save them time and provide comprehensive documentation for each algorithm.
Keep this in mind when developing a new algorithm.

There are multiple ways to contribute, the most common ones are:

    * New algorithm
    * Documentation
    * Examples
    * Bug reports

We encourage contribution of algorithms that are compliant with the |charts_link| calculations engine. Therefore, this
guide focuses on the requirements to comply with it. Nevertheless, we accept any other algorithms (not exposed through
|charts_link|) to be used by installing the python package in your preferred development environment.

*Although the core of this project are the industrial algorithms, improving our documentation is very
important and making our library more robust over time is of paramount importance. Please don't hesitate to submit a
Github pull request for something as small as a typo.*

Open source contributions
=========================

Thank you for considering contributing to InDSL! We welcome all contributions as listed above.
We encourage you to read this document to understand how to contribute to the project.
Also, we are happy to help you get started, and we welcome your efforts to improve InDSL 
as long as everyone involved is treated with respect. Cordiality is highly appreciated. 
Please read our `Code of Conduct <https://indsl.docs.cognite.com/code_of_conduct.html>`_ before contributing.

A good PR should be concise, clear, and easy to understand. In order to contribute, follow these steps:


1. **Fork the repository**: Fork the `repository <https://github.com/cognitedata/indsl>`_ 
to your own GitHub account.

2. **Run the tests**: Confirm that the tests pass on your local machine. We use `pytest` for testing. 
If they fail and you are unable to fix the issue, please reach out to us.

3. **Make your changes**: Make your changes to the code base. Make sure to follow the coding style and documentation guidelines.
Pre-commit checks will run automatically when you push your changes.
You can also run pre-commit checks manually for all staged files by running ``poetry run pre-commit run --all-files``. 
We follow the Google Python Style Guide for docstrings.

4. **Write tests**: If you are adding a new feature or fixing a bug, write tests using the `pytest` framework to cover the new code. 
Make sure that they pass.

5. **Make a pull request**: Once you are satisfied with your changes and all of the tests pass, make a pull request 
in the base repository using the conventional commit message format.


Code Review Process
-------------------

Contributions will only be merged after a code review. You are expected to address and incorporate feedback from the review unless 
there are compelling reasons not to. 
If you disagree with the feedback, present your objections clearly and respectfully. 
If the feedback is still deemed applicable after further discussion, you must either implement the suggested changes or choose 
to withdraw your contribution.

Documentation Contributions
---------------------------

Improvements to our documentation are much appreciated! The documentation source files are located in the 
`docs-source/source <https://github.com/cognitedata/indsl/tree/main/docs-source/source>`_ directory of 
our codebase. They are formatted in `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
and compiled with Sphinx to produce comprehensive documentation.

Contributing a new Cognite Charts compliant algorithm
=====================================================

For an algorithm to play well with the Charts front-end (user interface) and the
calculations back-end it has to adhere to some function input and output requirements, documentation (docstrings) format and a few
other requirements to expose the algorithm to the front and back-end. The first few basic requirements to keep in mind
before developing and algorithm are:

    1. It must belong to a particular toolbox. All the toolboxes are listed under the ``indsl/`` folder.
    2. It must be a python function: ``def():``
    3. Input data is passed to each algorithm as one or more ``pd.Series`` (one for each time series) with a ``datetime`` index.
    4. The output must be a ``pd.Series`` with a ``datetime`` index for it to be displayed on the UI.
    5. Function parameters types allowed are:

        * Time series: ``pd.Series``
        * Time series or float: ``Union[pd.Series, float]``
        * Integer: ``int``
        * Float: ``float``
        * `Enumerations <https://docs.python.org/3/library/enum.html>`_: ``Enum``
        * String: ``str``
        * Timestamp: ``pd.Timestamp``
        * Timedelta: ``pd.Timedelta``
        * String option: ``Literal``
        * List of floats: ``List[int]``
        * List of floats: ``List[float]``
        * Optional type: ``Optional[float]``

Functions exposed to Charts should also contain the following: 
   1. Steps used in the calculation:
      A detailed explanation of the algorithm or formula used in the function must be provided. For example:
      .. code:: python

         r"""Total head calculation.
         ...
         Steps:
         1. Subtract the suction pressure from the discharge pressure to find the pressure difference.
         2. Divide the result by the product of the gravitational constant (9.81 m/s²) and the fluid density.
         3. The result is the total head (in meters) of the pump.
         ...
         """

   2. URL links to further external documentation
      If your function uses an external formula or concept that requires further explanation, 
      you can link to external documentation for additional context. Here is how to add links:

      .. code:: python

         r"""
         ...
         For more information on centrifugal pumps and head pressure, you can visit:
         `Centrifugal Pump Theory <https://www.pumpfundamentals.com/>`_.
         ...
         """

   3. Main formulas, in Latex format, used for the calculations
      Make sure to use the proper LaTeX formatting in the docstring 
      to clearly describe the mathematical operations involved. Here’s an example:

      .. code:: python

         r"""
         ...
         Formula for total head :math:`h` [m]:
         .. math::
            h = \frac{P_{discharge} - P_{suction}}{9.81 \cdot \rho_L}

         Where:
         - :math:`P_{discharge}` is the discharge pressure [Pa]
         - :math:`P_{suction}` is the suction pressure [Pa]
         - :math:`\rho_L` is the density of the fluid [:math:`kg/m^3`].
         ...
         """
   4. Good description of what each input parameter does and its limits.
      Each input parameter should be described in detail, including units, data types, 
      and possible value ranges. Here's an example of how to describe parameters with clear limits:

      .. code:: python

         r"""
         ...
         Args: discharge_pressure: The discharge pressure of the pump in Pascals [Pa].
            This value can either be a float or a time series (`pd.Series`). It must be positive and greater than the suction pressure.
         suction_pressure: The suction pressure of the pump in Pascals [Pa].
            This value can either be a float or a time series (`pd.Series`). It should be positive, and typically less than the discharge pressure.
         den: The density of the fluid in kilograms per cubic meter [:math:`kg/m^3`].
            This value must be a positive float or time series. A typical value for water is 1000 kg/m^3.
         align_timesteps: Boolean flag to align time steps of the input time series.
            If set to True, the function will automatically align the input time series by their timestamps. Default is False.
         ...
         """

.. note::

    We currently support python functions with ``pd.Series`` as the type of data input and outputs. This restriction
    is in place to simplify how the Charts infrastructure fetches and displays data.


Preliminaries and setup
-----------------------

.. note::

    Avoid duplicating code. Before starting a new algorithm, check for similar ones in the following places:
        * The `toolboxes <https://github.com/cognitedata/indsl/tree/main/indsl>`_
        * The `PR list <https://github.com/cognitedata/indsl/pulls>`_

This project uses `Poetry <https://python-poetry.org/>`_ for dependency management. Install it before starting

.. prompt:: bash $

   pip install poetry


1. For open source contributions, fork the `InDSL <https://github.com/cognitedata/indsl>`_ main repository on
   GitHub to your local environment. If the contribution is internal, you may clone the repository directly.

.. prompt:: bash $

    git clone git@github.com:cognitedata/indsl.git
    cd indsl

2. Install the project dependencies.

.. prompt:: bash $

    poetry install --all-extras

3. Synchronize your local main branch with the remote main branch.

.. prompt:: bash $

    git checkout main
    git pull origin main

Develop your algorithm
----------------------

1. Create a feature branch to work on your new algorithm. Never work on the *main* or *documentation* branches.

   .. prompt:: bash $

      git checkout -b my_new_algorithm

2. Install *pre-commit* to run code style checks before each commit.

   .. prompt:: bash $

      poetry run pre-commit install  # Only needed if not installed
      poetry run pre-commit run --all-files

3. If you need any additional module not in the installed dependencies, install it using the ``add`` command. If you
   need the new module for development, use the ``--dev`` option:

   .. prompt:: bash $

      poetry add new_module

   .. prompt:: bash $

      poetry add new_module --dev

4. Develop the new algorithm on your local branch. Use the exception classes defined in
   `indsl/exceptions.py <https://github.com/cognitedata/indsl/tree/main/indsl/exceptions.py>`_
   when raising errors that are caused by invalid or erroneous user input. InDSL provides the @check_types
   decorator (from `typeguard <https://github.com/agronholm/typeguard>`_) for run-time type checking,
   which should be used instead of checking each input type explicitly. When finished or reach an important
   milestone, use ``git add`` and ``git commit`` to record it:

   .. prompt:: bash $

       git add .
       git commit -m "Short but concise commit message with your changes"


   If your function is not valid for certain input values, an error **must** be thrown. For example,

   .. code-block:: python

       def area(length: float) -> float:
           if length < 0:
               raise UserValueError("Length cannot be negative.")
           return length**2


5. As you develop the algorithm it is good practice to add tests to it. All tests are stored in the root folder
   `tests/ <https://github.com/cognitedata/indsl/tree/main/tests>`_ using the same folder structure
   as the ``indsl/`` folder. We run ``pytest`` to verify pull requests before merging with the main
   version. Before sending your pull request for review, make sure you have written tests for the algorithm and ran
   them locally to verify they pass.

.. note:: **New algorithms without proper tests will not be merged - help us keep the code coverage at a high level!**

Core or Extras
--------------

InDSL is divided into two main categories: core and extras. The core algorithms are the ones that only require
``numpy``, ``scipy``and ``pandas`` as dependencies. The extras are algorithms that require additional dependencies.

If your algorithm requires additional dependencies, add them to the ``pyproject.toml`` file as optional dependencies and
also add them under the ``tool.poetry.extras`` section in an appropriate category. The dependencies will also need to be
lazy loaded to avoid loading them when the core part of the library is imported. To do this you need to import the
dependencies in the function itself, and not at the top of the file.


Document your algorithm
-----------------------

Charts compliant algorithms must follow a few simple docstrings formatting requirements for the information to be parsed
and properly displayed on the user interface and included in the technical documentation.

1. Use `r"""raw triple double quotes"""` docstrings to document your algorithm. This allows using backslashes in the
   documentation, hence LaTeX formulas are properly parsed and rendered. The documentation targets both data science
   developers and Charts users and the `r"""` allows us properly render formulas in the Charts UI and
   in the InDSL documentation. If you are not sure how to document, refer to any algorithm in the
   ``indsl``/ folder for inspiration.

2. Follow `Google Style  <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_ unless otherwise is stated in this guide.

3. **Function name**: after the first `r"""`, write a short (1-5 words) descriptive name for your function with no punctuation at the end.
   This will be the function name displayed on the Charts user interface.

4. Add an empty space line break after the title.

5. Write a comprehensive description of your function. Take care to use full words to describe input arguments.
   For example, in code you might use ``poly_order`` as an argument but in the description use ``polynomial order``
   instead.

6. **Parameter names and descriptions**: define all the function arguments after ``Args:`` by listing all arguments,
   using tabs to differentiate each one and their respective description. Adhere as close as possible to the following formatting rules for each parameter name and description:

    * A parameter name must have 30 characters or less, excluding units defined within square brackets ``[]``
      (more on this below). Square brackets are only allowed to input units in a parameter name. Using brackets within
      a parameter name for something different to units might generate an error in the pre-commit tests.
    * Must end with a period punctuation mark ``.``
    * Use LaTeX language for typing formulas, if any, as follows:

        * Use the command ``:math:`LaTeX formula``` for inline formulas
        * Use the command ``.. math::`` for full line equations

    * If a parameter requires specific units, these must be typed as follows:

        * Enclosed in square brackets ``[]``
        * In Roman (not italic) font
        * If using LaTeX language, use the ``:math:`` inline formula command, and the command ``\mathrm{}`` to render
          the units in Roman font.
        * Placed at the end of the string

      For example:

.. code:: python

   r"""
   ...
   Args:

       ...

       pump_hydraulic_power: Pump hydraulic power [W].
       pump_liquid_flowrate: Pump liquid flowrate [:math:`\mathrm{\frac{m^3}{h}}`].

       ...

This is a `basic example <https://github.com/cognitedata/indsl/blob/main/indsl/smooth/savitzky_golay.py>`_
of how to document a function :

.. code:: python

    r"""
    ...

    Args:
        data: Time series.
        window_length: Window.
            Point-wise length of the filter window (i.e. number of data points). A large window results in a stronger
            smoothing effect and vice-versa. If the filter window length is not defined by the user, a
            length of about 1/5 of the length of time series is set.
        polyorder: Polynomial order.
            Order of the polynomial used to fit the samples. Must be less than the filter window length.
            Hint: A small polynomial order (e.g. 1) results in a stronger data smoothing effect.
            Defaults to 1, which typically results in a smoothed time series representing the dominating data trend
            and attenuates fluctuations.

    Returns:
        pd.Series: Time series
        If you want, it is possible to add more text here to describe the output.

    ...
    """

7. Define the function output after ``Returns:`` as shown above.

8. The above are the minimal requirements to expose the documentation on the user interface and technical docs. But
   feel free to add more `supported sections <https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_.

9. Go to the ``docs-source/source/`` folder and find the appropriate toolbox ``rst`` file (e.g. ``smooth.rst``)

10. Add the a new entry with the name of your function as a subtitle, underlined with the symbol ``^``.

11. Add the sphinx directive ``.. autofunction::`` followed by the path to your new algorithm (see the example below).
    This will autogenerate the documentation from the code docstrings.

.. prompt:: text

    .. autofunction:: indsl.smooth.sg

11. If you have coded an example, add the sphinx directive ``.. topic:: Examples:`` and below it the sphinx reference
    to find the autogenerated material (see example below). The construct is as follows,
    ``sphx_glr_autoexamples_{toolbox_folder}_{example_code}.py``

.. prompt:: text

    .. topic:: Examples:

        * :ref:`sphx_glr_auto_examples_smooth_plot_sg_smooth.py`

Front and back end compliance
-----------------------------

For the  algorithm to be picked up by the front and back end, and display user relevant information, take the following
steps.

1. Add human readable names to each input parameter (not the input data) in your algorithm. These will be displayed on
   the UI, hence avoid using long names or special characters.

2. Add a technical but human readable description of your algorithm, the inputs required, what it does, and the
   expected result. This will be displayed on the UI and targets our users (i.e. domain experts).

    .. todo:: Implement the human readable description and input variable names for the algorithms

3. Add the @check_types decorator to the functions that contain Python type annotations. This makes sure that the function is always called with inputs of the same type as specified in the function signature.

4. Add your function to the attribute ``__cognite__`` in the ``__init__.py`` file of the toolbox module your algorithm belongs to. For example, the
    `Savitzky-Golay smoother
    <https://github.com/cognitedata/indsl/blob/main/indsl/smooth/savitzky_golay.py>`_
    (:meth:`indsl.smooth.sg`) belongs to the ``smooth`` toolbox. Therefore, we add ``sg`` to the list ``__cognite__``
    in the file ``indsl/smooth/__init__.py``.

This would be a good time to push your changes to the remote repository

Add an example to the Gallery of Charts
---------------------------------------

:ref:`sphx_glr_auto_examples` is an auto generated collection of examples of our industrial data science
algorithms. Following the steps below, your example will be automatically added to the gallery. We take care of
auto generating the figures, adding the code to the gallery, and links to downloadable python and notebook versions
of your code for other data scientists to use or get inspired by (sharing is caring!). We use `Sphinx-Gallery
<https://sphinx-gallery.github.io/stable/index.html>`_ for this purpose, if you want to find out more about what you
can do to generate generate your example.

We want to offer our user and developers as much information as possible about our industrial algorithms. Therefore we
**strongly encourage** all data scientist and developers to include one or more examples (license to go crazy here)
to show off all the amazing features and functionalities of your new algorithm and how it can be used.

1. For open source contributions, fork the INDSL repo and create your own local branch. 
   For internal contributions, you may clone the repository directly.
2. Go to the toolbox folder in ``examples`` where your algorithm belongs to (e.g. ``smooth``)
3. Create a new python file with the prefix *plot_*. For example ``plot_my_new_algo_feature.py``.
4. At the top of the file, add a triple quote docstring that start with the title of your example enclose by
   top and bottom equal symbols (as shown below), followed by a description of your example. For inspiration, check
   the :ref:`sphx_glr_auto_examples` or one of the examples in the repository
   (e.g. ``examples/smooth/plot_sg_smooth.py``).

.. prompt:: python

    """
    =============
    Example title
    =============
    Description of the example and what feature of the algorithm I'm showing off.
    """

    import pandas as pd
    ...

5. Once you are done developing the example record your changes using ``git add <path_to_file>``, ``git commit -m <commit_message>`` and ``git push -u origin <your_branch_name>``
6. You can test the Sphinx build of your PR by following the steps in the section below.

Verify documentation build
--------------------------

It is highly recommended to check that the documentation for your new function is built and displayed
correctly. Note that you will need all of the following Sphinx python libraries to successfully build the documentation (these packages can be installed with pip):
* sphinx-gallery
* sphinx
* sphinx-prompt
* sphinx-rtd-theme

While testing the build, some files that *should not be committed to the remote repository*, will be
autogenerated in the folder ``docs-source/source/auto_examples/``. If these are committed nothing will really happen,
except for the PR probably being longer than expected and could confuse the reviewers if they are not aware of this.
To avoid it there are two two options:

1. Don't stage the files inside the folder ``docs-source/source/auto_examples/``, or
2. add the folder ``docs-source/source/auto_examples/`` to the file ``.git/info/exclude`` to locally exclude the folder
   from any commit. You can use your IDE git integration to locally exclude files
   (e.g. `PyCharm <https://www.jetbrains.com/help/pycharm/set-up-a-git-repository.html#ignore-files>`_).

Once you taken care of the above, do the following:

1. Install the dependencies needed to build the documentation:

.. prompt:: terminal

    poetry install --with docs

2. In your terminal, go to the folder ``docs-source/``
3. Clean the previous build (if any) using

.. prompt:: terminal

    make clean

4. Build the documentation with

.. prompt:: terminal

   poetry run make html

5. If there were errors during the build, address them and repeat steps 2-3.

6. If the build was successful, open the html file located in `build/html/index.html` and review it navigating to the
   section(s) relevant to your new function.

   For mac users the file can be opened with the following command:

.. prompt:: terminal

    open build/html/index.html


7. Once satisfied with the documentation, commit and push the changes.


Version your algorithm
----------------------

.. note::
      This section is only relevant if you are changing an existing function in InDSL.

For industrial applications, consistency and reproducibility of calculation results is of critical importance.
For this reason, InDSL keeps a version history of InDSL functions that developers user can choose from.
Older versions can be marked as deprecated to notify users that a new version is available.
The example :ref:`sphx_glr_auto_examples_versioning_versioned_function.py` demonstrates in more detail how the function versioning works in InDSL.

Do I need to version my algorithm?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You need to version your algorithm if:

1) You are changing an existing InDSL function, and one of the following conditions holds:

   * The signature of the new function is incompatible with the old function. For instance if a parameter was renamed or a new parameter was added without a default value.
   * The modifications change the function output for any given input.
2) You are changing a helper function that is used by other InDSL functions. In that case you need to version the helper function and all affected InDSL functions.

.. note::
        In order to avoid code duplication, one should explore if the modifications can be implemented in a backwards-compatible manner (for instance through a new parameter with a default value).


How do I version my function?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As an example, we consider a function `myfunc` in `mymod.py`.
A new function version is released through the following steps.

1) Move the function from `mymod.py` to `mymod_vX.py`, where `X` denotes the current function version. If the function is not versioned yet, create the file `mymod_v1.py`.
2) If not already present, add the :func:`versioning.register` decorator to the function. Specifically,

   .. code-block:: python

           # file: mymod_v1.py
           @check_types
           def myfunc(...)
              # old implementation

   becomes:

   .. code-block:: python

           # file: mymod_v1.py
           from indsl import versioning

           @versioning.register(version="1.0", deprecated=True)
           @check_types
           def myfunc(...)
              # old implementation

   **Note**: The first version of any function **must** be 1.0! Also note that :code:`deprecated=True`: InDSL allows at most
   one non-deprecated version. For functions already in Charts, deprecating all versions will remove the functions from the front-end.

   **Note**: `check_types` decorator should be placed before `versioning.register` decorator.

3) Add the new implementation to `mymod.py` and import `mymod_v1.py`. The modified `mymod.py` file will look like:

   .. code-block:: python

           # file: mymod.py
           from indsl import versioning
           from . import mymod_v1  # noqa

           @versioning.register(version="2.0", changelog="Describe here how the function changed compared to the previous version")
           def myfunc(...)
              # new implementation

   Make sure to increment the version number (a single positive integer) of the new implementation. Optionally, non-breaking changes can be versioned.
   In that case follow the `semantic versioning guidelines <https://semver.org/>`_.

4) Make sure the all versions of the function `myfunc` are tested. If the tests of the most recent version are in `test_mymod.py`, tests for the deprecated
   function can be placed in `test_mymod_v1.py`.


Create a pull request
---------------------

Before a PR is merged it needs to be approved by of our internal developers. If you expect to keep on working on your
algorithm and are not ready to start the review process, please label the PR as a ``draft``.

To make the review process a better experience, we encourage complying with the following guidelines:

1. Give your pull request a helpful title. If it is part of a `JIRA task in our development backlog
   <https://cognitedata.atlassian.net/jira/software/projects/PI/boards/402/backlog>`_, please add the task reference so
   it can be tracked by our team. If you are fixing a bug or improving documentation, using "BUG <ISSUE TITLE>" and
   "DOC <DESCRIPTION>" is enough.

2. Make sure your code passes all the tests. You could run ``pytest`` globally, but this is not recommended as it
   will take a long time as our library grows. Typically, running a few tests only on your new algorithm is enough.
   For example, if you created a ``new_algorithm`` in the ``smooth`` toolbox and added the tests
   ``test_new_algorithm.py``:

   * ``pytest tests/smooth/test_new_algorithm.py`` to run the tests specific to your algorithm
   * ``pytest tests/smooth`` to run the whole tests for the ``smooth`` toolbox module

    .. todo:: Add pytest example for single function documentation
    .. todo:: Add pytest example for building single function Gallery documentation

3. Make sure your code is properly commented and documented. We can not highlight enough how important documenting
   your algorithm is for the succes of this product.

4. Make sure the documentation renders properly. For details on how to build the documentation. Check our documentation guidelines (WIP). The official documentation will be built and deployed by our CI/CD workflows.

5. Make sure the function renders properly in the UI.
   To preview the function node access the storybook build results url, which can be found in the PR comments.
   In chromatic, scroll down and inspect the stories for the function.

6. Add test to all new algorithms or improvements to algorithms. These test add robustness to our code base and
   ensure that future modifications comply with the desired behavior of the algorithm.

7. Run ``black`` to auto-format your code contributions. Our pre-commit will run black for the entire project once you
   are ready to commit and push to the remote branch. But this can take some time as our code base grows. Therefore, it
   is good practice to run periodically run ``black`` only for your new code.

.. prompt:: bash

    black {source_file_or_directory}

This is not an exact list of requirements or guidelines. If you have suggestions, don't hesitate to submit an issue or
a PR with enhancement to this document.

Finally, once you have completed your new contribution, sync with the remote/main branch one last in case there have
been any recent changes to the code base:

.. prompt:: bash

    git checkout main
    git pull
    git checkout {my_branch_name}
    git merge main

Then use ``git add``, ``git commit``, and ``git push`` to record your new algorithm and send it to the remote
repository:

.. prompt:: bash

    git add .
    git commit -m "Explicit commit message"
    git push

Go to the `InDSL repository PR page <https://github.com/cognitedata/indsl/pulls>`_, start
a ``New pull request`` and let the review process begin.


.. |charts_link| raw:: html

   <a href="https://charts.cogniteapp.com/" target="_blank">Cognite Charts</a>

.. |charts_docs| raw:: html

   <a href="https://docs.cognite.com/cdf/charts/guides/getting_started.html" target="_blank">Charts documentation page</a>

Contributing a free form algorithm
=============================================
It is possible to contribute to InDSL without the algorithm being exposed in the Charts application.
In this case, the algorithm will only be available to users who install the InDSL python package.
It  **should not** be included in the ``__cognite__`` attribute of the toolbox `__init__.py` file.
Although the algorithm doesn't need to meet the requirements mentioned in the :ref:`previous <contributing-a-new-charts-compliant-algorithm>` section, it is still important to
 document it properly, add all necessary tests and potentially an example to the documentation.

Coding Style
============

To ensure consistency throughout the code, we recommend using the following style conventions when contributing to the library:
    * Call the time series parameter of your function ``data`` unless a more specific name can be given, like ``pressure`` or ``temperature``.
    * Use abbreviations when defining the types of function arguments. For example ``pd.`` instead of ``pandas``.

Reviewer guidelines

Any InDSL function that is exposed in the Charts application (i.e. any function that is listed in `__cognite__` in the `__init__.py` files), must be reviewed by a member of the Charts development team.
