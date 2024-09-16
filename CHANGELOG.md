## v8.6.0 (2024-09-16)

### Feat

- exposing Sustainability package  (#313)
- more dimensionless numbers and pipe friction models (#273)

### Fix

- Bugfix for the mock scatter plot (#315)

## v8.5.0 (2024-09-09)

### Feat

- functions operating on all elements in the time series (#274)
- interpolation based on sequence data (#277)
- mock scatter plot functionality (#279)

### Fix

- Add example units to guide users (#253)
- **deps**: update dependency scikit-image to ^0.24.0 (#215)
- **deps**: update dependency numba to ^0.60.0 (#261)
- disable oscillation detection in charts (#240)
- **deps**: update dependency scikit-image to ^0.22.0 (#42)

## v8.4.0 (2024-03-22)

### Feat

- Add Discharge Reciprocating Pump function

## v8.3.3 (2024-03-20)

### Fix

- change  descriptions of arguments
- **deps**: update dependency numba to ^0.59.0

## v8.3.2 (2024-02-23)

### Fix

- generation of parameter name should be formatted
- **deps**: update dependency pandas to ~2.2.0
- **deps**: update dependency scikit-learn to v1.4.0
- **deps**: update dependency numpy to v1.26.3

## v8.3.1 (2024-02-02)

### Fix

- github actions script to push json to locize
- script to add docstring info in a json file

## v8.3.0 (2024-01-15)

### Feat

- python-3.12 (#6)

### Fix

- removing recommended python version
- Switch back function descriptions (#72)
- Update note about availability after open sourcing (#74)
- upgrade pandas to 2.1.4 (#69)
- **deps**: Revert "fix(deps): update dependency pandas to ~2.1.0 (#18)"
- **deps**: update dependency pandas to ~2.1.0 (#18)

## v8.2.1
* chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.8.0 in #1
* chore(deps): lock file maintenance in #3
* chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.8.1 in #4
* chore(deps): update pre-commit hook psf/black to v23.9.0 in #5
* chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.8.2 in #7
* fix: pandas future (deprecation) warnings  in #9
* chore(deps): update pre-commit hook psf/black to v23.9.1 in #12
* chore: deprecate support for python 3.8 in #10
* fix(deps): update dependency numpy to v1.25.2 in #14
* docs: rename master with main  in #19
* docs: change support email  in #17
* chore(deps): update dependency ruff to ^0.0.288 in #20
* chore(deps): update pre-commit hook astral-sh/ruff-pre-commit to v0.0.288 in #21
* docs: update compiled doc files  in #22
* chore(deps): update dependency ruff to ^0.0.289 in #23
* chore(deps): update pre-commit hook astral-sh/ruff-pre-commit to v0.0.289 in #24
* fix: cannot find function version in #26
* chore: remove python 3.8 for CI test extras  in #29
* docs: build documentation  in #28
* chore(deps): update pre-commit hook pre-commit/pre-commit-hooks to v4.5.0 in #16
* chore(deps): update pre-commit hook pre-commit/mirrors-mypy to v1.6.0 in #15
* chore(deps): update dependency docstring-to-markdown to v0.12 in #13
* fix: corrected types by @anvar-akhiiartdinov in #31
* chore(deps-dev): bump urllib3 from 2.0.6 to 2.0.7  in #27
## v8.2.0
* fix(deps): update dependency numpy to v1.25.1 in #1056
* chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.5.4 in #1058
* fix(deps): update dependency scipy to v1.11.1 in #1057
* fix: raise UserValueError when snr_db is above certain value for white_noise in #1049
* fix: Integer division or modulo by zero in _make_index() in #1036
* fix: check if wvalve changes to avoid IndexError in #1052
* test: remove jit compile of tests in #1059
* chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.6.0 in #1060
* chore(deps): lock file maintenance in #1061
* docs: update docs in #1055
* fix(deps): update dependency numpy to v1.25.2 in #1062
* chore(deps): update pre-commit hook pre-commit/mirrors-mypy to v1.5.0 in #1065
* chore(deps): lock file maintenance in #1064
* chore(deps): update dependency sphinx to <7.3 in #1067
* chore(deps): update pre-commit hook pre-commit/mirrors-mypy to v1.5.1 in #1068
* fix(deps): update dependency scipy to v1.11.2 in #1063
* chore(deps): update dependency ruff to ^0.0.285 in #1069
* chore(deps): update pre-commit hook astral-sh/ruff-pre-commit to v0.0.285 in #1070
* chore(deps): lock file maintenance in #1071
* chore(deps): update dependency sphinx-gallery to ^0.14.0 in #1072
* chore(deps): update dependency sphinx to v7.2.2 in #1074
* chore(deps): lock file maintenance in #1076
* chore(deps): lock file maintenance in #1077
* chore(deps): update dependency ruff to ^0.0.286in #1078
* chore(deps): update pre-commit hook astral-sh/ruff-pre-commit to v0.0.286 in #1079
* chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.7.0 in #1081
* chore(deps): update dependency sphinx to v7.2.3 in #1080
* chore: add cognite copyright 2023 to all source files in #1082
* fix(deps): update dependency pandas to v2.1.0 in #1085
* chore(deps): lock file maintenancein #1086
* docs: simplify-docs-build in #1087
* chore(deps): update dependency ruff to ^0.0.287 in #1092
* chore(deps): update dependency sphinx to v7.2.5 in #1089
* chore(deps): update pre-commit hook astral-sh/ruff-pre-commit to v0.0.287 in #1093
* refactor: clean up comments [AH-1903] in #1088
* feat: export charts specific functions [AH-1067] in #1083

## v8.1.0
### Feat
* chore(deps): update dependency sphinx to <7.2 (#1051)
* chore(deps): update python docker tag to v3.11.5 (#1053)
* chore(deps): lock file maintenance (#1054)
* ### Fix
* fix: Add type conversion in Change Point Detection algorithm (#1047)
* fix: check if valve is empty (#1050)
## v8.0.1
### Fix
* fix: create a poetry group for fluids (#1048)
## v8.0.0
### Feat
* docs: build docs (#1021)
* chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.4.0 (#1023)
* chore(deps): update dependency ruff to ^0.0.275 (#1025)
* chore(deps): lock file maintenance (#1026)
* chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.5.0 (#1027)
* chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.5.1 (#1028)
* chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.5.2 (#1029)
* chore(deps): lock file maintenance (#1030)
* chore: Bump word-wrap from 1.2.3 to 1.2.4 in /.function-preview (#1022)
* Bump tough-cookie from 4.1.2 to 4.1.3 in /.function-preview (#1010)
* Bump semver from 5.7.1 to 5.7.2 in /.function-preview (#1011)
* chore(deps): update pre-commit hook pre-commit/mirrors-mypy to v1 (#1008)
* chore(deps): update pre-commit hook astral-sh/ruff-pre-commit to v0.0.275 (#1024)
* chore(deps): update dependency ruff to ^0.0.276 (#1031)
* chore(deps): update dependency ruff to ^0.0.277 (#1034)
* chore(deps): update dependency sphinx to v7 (#1033)
* chore(deps): lock file maintenance (#1035)
* chore(deps): update pre-commit hook psf/black to v23.7.0 (#1038)
* refactor: InDSL Core and Extras Split (#1000)
* chore(deps): update dependency ruff to ^0.0.278 (#1041)
* chore(deps): lock file maintenance (#1042)
* chore(deps): update dependency ruff to ^0.0.284 (#1043)
* chore(deps): update pre-commit hook astral-sh/ruff-pre-commit to v0.0.278 (#1032)
* chore(deps): update pre-commit hook commitizen-tools/commitizen (#1044)
* chore(deps): lock file maintenance (#1045)
### Fix

* fix(deps): update dependency scipy to v1.11.1 (#1039)
* fix: raise UserValueError if alpha is outside the range (0, 1] in Cusum (#1037)

## v7.0.1
### Fix
fix: remove matplotlib from indsl main depenencies, use lazy import (#1019)

## v7.0.0
### Feat

- chore(deps): update pre-commit hook commitizen-tools/commitizen to v3.3.0 (1014)
- chore(deps): lock file maintenance (#1016)
- chore(deps): update dependency myst-parser to v2 (#1015)

### Fix
- fix: exclude plot from oscillation detector in charts (#1017)
## v6.6.0a1 Pre-release

### Feat
- chore(deps): lock file maintenance (#1009)
- chore(deps): update dependency sphinx to v7 (#1006)

### Fix
- fix: raise UserValueError if x is duplicated (#1005)
- fix: updates after panadas v2 (#1012)

## v6.6.0a0 (2023-07-04)

### Feat

- calculate datapoint difference over a time period (#851)
- valve flow for compressible fluid (#670)

### Fix

- use Literal from typing_extensions (#999)
- **deps**: update dependency scikit-image to ^0.21.0 (#1003)
- **deps**: update dependency pandas to v2 (#959)
- **deps**: update dependency typeguard to v4 [DEGR-2625] (#988)
- **deps**: update dependency numpy to <1.24.4 (#991)
- **deps**: update dependency scipy to v1.10.1 (#973)
- **deps**: update dependency numba to ^0.57.0 (#980)
- **deps**: update dependency statsmodels to ^0.14.0 (#982)
- add automerge minor to renovate config (#966)
- remove auto-update in renovate config (#964)
- **deps**: update dependency scikit-image to ^0.20.0 (#951)
- resolve precommit errors

## v6.5.0 (2023-03-31)

### Feat
- feat: add groupby region calculation (#933)

### Fix

- add `pd.Timedelta` validation to `trapezoidal_integration` (#946)
- move away from numba implementation for fluids library (#943)

## v6.4.4 (2023-03-23)
### Fix

- remove emd (#940)

## v6.4.3 (2023-02-23)

### Fix

- **deps**: update dependency packaging to v23 (#921)
- remove emd dependency from indsl (#923)

## v6.4.2 (2023-01-24)

### Fix

- numpy warning (#909)

## v6.4.1 (2023-01-24)

### Fix

- data quality base class handle consecutive gaps as separate events [DEGR-1298] (#906)
- **deps**: update dependency packaging to v22 (#898)
- interpolation typo (#886)

## v6.4.0 (2022-12-13)

### Feat

- add drilling toolbox and basic detection algorithms (#832)
- check licenses with dependencies CI action [DEGR-956] (#879)

### Fix

- remove gustavo as author (#884)
- Refactor title of example for value decrease check (#880)
- solved not working link
- update pre-commit flake url to github instead of gitlab (#872)

## v6.3.1 (2022-11-09)

### Fix

- **publish**: add auto-merge label to PR (#863)

## v6.3.0 (2022-11-08)

### Feat

- add new optional argument for resample_timeseries utility function (#830)
- add auto approve workflow for renovate[bot] PRs (#854)
- add auto update label to renovate (#841)

### Fix

- **deps**: update dependency numba to ^0.56.0 (#827)
- Instantiate outlier detection function (#846)

## v6.2.0 (2022-09-29)

### Feat

- Data Quality Function for validity dimension - Out of range outlier detection (#667)

### Fix

- **deps**: update dependency kneed to ^0.8.0 (#831)

## v6.1.0 (2022-08-25)

### Feat

- add data profiling metrics (#739)

## v6.0.1 (2022-08-17)

### Fix

- fix failing pypi push (#819)

## v6.0.0 (2022-08-17)

### Feat

- add range option to remove function (#814)

### Fix

- print correct filenames in docstring test (#808)
- smoothers should handle empty data series [CHART-1005] (#800)
- raise UserValueError when magnitude < 0 in perturb_timestamp [CHART-1031] (#804)
- Change unit of timedelta to milliseconds instead of seconds (#795)
- ensure that functions with old naming scheme (e.g. WAVELET_FILTER) are copy of v1.0 (#792)
- Handle empty time series in gas_density function [CHART-1090] (#805)
- use pd.Timedelta type in smoother functions [CHART-1030] (#801)
- reindex returns pd.Series instead of List[pd.Series] [CHART-1344] (#785)
- use pd.Timedelta datatypes in numerical_calculus functions [CHART-973] (#797)
- remove invalid wavelet sym1 and add test [CHART-905] (#796)
- **deps**: pin dependencies (#773)

### Refactor

- remove unused private function _get_sample_frequency (#809)

## v4.8.2 (2022-08-09)

### Fix

- all versioned functions start with v1.0 (#788)

## v4.8.1 (2022-08-09)

### Refactor

- introduce new naming convention for versioned function (#782)

### Fix

- replace Enum with Literal types in InDSL functions [CHART-763] [CHART-1007] (#781)
- use stricter data types in remove outliers [CHART-1342] (#776)
- deprecate forecasting functions [CHART-1257] (#778)
- increase robustness and performance of outlier removal function [CHART-1264] (#774)
- use numpy's polynomial package for poly fitting/evaluation [CHART-1295] (#765)
- remove infinity values from polynomial regression [CHART-1282] (#766)
- use stricter data types for drift function [CHART-1281] (#767)
- interpolate handles array inputs of length 1 [CHART-1339] (#761)
- RuntimeError -> UserValueError in steady state detector function [CHART-1296] (#763)
- use stricter datatypes in interpolate function [CHART-1322] (#762)
- **deps**: update dependency @testing-library/react to v13 (#756)
- added type hinting to functions [CHARTS-1338] (#751)

## v4.8.0 (2022-07-26)

### Fix

- Increase coverage for confidence bands function (#746)
- Added coverage for drift detector (#745)
- fixed validtation tests in alma.py [CHARTS-1332] (#744)
- raise UserValueError for eps parameter and add error test (#735)
- cusum docstring (#732)

### Feat

- Refactor outlier removal and create function for outlier detection (#741)
- Pearson correlation rolling window (#711)

## v4.7.0 (2022-07-07)

### Fix

- Add duration unit for unchanged signal detector (#712)
- fix example recycle valve power loss (#717)

### Feat

- python 3.10 (#721)

### Refactor

- remove todo functions (#707)

## v4.6.0 (2022-06-16)

### Feat

- add cusum function (#540)
- add sustainability calculations (#686)

### Fix

- coverage upgrade for butterworth filter (#690)

## v4.5.1 (2022-06-09)

### Fix

- error message in holt winters predictor (#688)
- make test of initial version accept more formats (#685)
- delete tab.py as it is not used nor covered by a test (#679)
- remove double tolerance on equipment unit tests (#673)
- seasonal_periods must be larger than 1 [CHART-1263] (#675)

## v4.5.0 (2022-05-25)

### Fix

- add correct type of parameters for density methods (DQ score) (#666)

### Feat

- Unchanged signal identification of time series (#638)

## v4.4.1 (2022-05-18)

### Fix

- typo in Pump recycle valve power description (#661)

## v4.4.0 (2022-05-11)

### Feat

- allow union of scalar and series parameters to pump and valve f… (#636)
- Rolling standard deviation of time delta (#582)

### Fix

- fix bug in reindex if bounded=True and input contains NaNs [CHART-1224] (#644)
- Raise UserValueError if time series is empty in remove_outliers [CHART-1226] (#642)

### Refactor

- restructure add equipment (#624)

## v4.3.1 (2022-05-04)

### Fix

- wrong resampling when one input has two values (#631)
- small bug (#627)
- constant value resolution (#625)

## v4.3.0 (2022-04-27)

### Fix

- add density functions to init (#616)
- Tuple to floats inputs (#613)
- fix typing annotation for GapDataQualityScoreAnalyser.compute_score (#614)
- throw UserValueErrors in ts_utils [CHART-1190] (#602)

### Feat

- [DataQuality] density function and score (#562)
- Constant value (#583)
- centrifugal pump recirculation energy loss (#586)

## v4.2.2 (2022-04-21)

### Fix

- gracefully handle empty time series in DataQualityScore.compute_score (#594)

## v4.2.1 (2022-04-21)

### Fix

- format flag time series  (#588)
- typo gas density calcs (#574)
- change to UserRuntimeError in calculate_compressibility [CHART-1182] (#587)
- remove helper function from __init__ (#579)

## v4.2.0 (2022-04-20)

### Fix

- sg from float to series (#569)

### Feat

- add uncertainty estimation data quality function [Chart-1102] (#535)
- feat: gap identification by timedelta threshold (#546)

## v4.1.1 (2022-03-30)

### Fix

- formatting centrifugal pumps (#548)

## v4.1.0 (2022-03-22)

### Fix

- function extreme (outlier detection) had an error in a formula.   (#541)
- fix crashing resample_to_granularity for interpolation aggregats [CHART-1103] (#537)
- skip auto alignment if indices are already aligned [CHART-1034] (#522)
- update extreme outlier logic (#513)

### Feat

- New function: centrifugal pump parameters (#530)
- New function: change to sdk-core to limit package size (#533)
- New function: trend extraction  [CHART-992] (#525)

### Refactor

- replace deprecated distutils.Version class with packaging.version.Version (#524)

## v4.0.2 (2022-03-01)

### Fix

- [DataQuality] Improvements to data quality score algorithms (#512)

## v4.0.1 (2022-02-28)

### Fix

- fix sg error (#507)
- [DataQuality] More robust gap based data quality score (#504)

## v4.0.0 (2022-02-24)

### Fix

- Versioning: replace deprecation_warning with changelog [CHART-1041] (#494)
- Versioning: enforce 1.0 as first version for versioned functions [CHART-1055] (#492)
- Fix argument docstring for well_prod_status function (#483)

### Feat

- DataQuality: add abstract and gap-based data quality scores [Chart-1023] (#474)
- New function: arithmetic mean for multiple time series (#481)
- New function: arithmetic mean function for two time series (#471)
- New function: sliding window integrator (#438)

### Refactor

- Collect common validation functions into validations.py (#459)

## v3.1.1 (2022-02-02)

### Fix

- Improve visual representation of change points (#393)

## v3.1.0 (2022-01-21)

### Feat

- Negative running hours data quality model  (#340)
- Threshold function [CHART-948] (#382)

### Fix

- Change RuntimeError to UserRuntimeError in change point detector [CHART-937] (#370)
- Improve input validation for trapezoidal_integration [CHART-935] (#371)
- Improve input validation for differentiate [CHART-938] (#369)
- Change RuntimeError to UserRuntimeError in arma predictor [CHART-934] (#372)
- Remove fluid tab parser (#355)

## v3.0.0 (2022-01-14)

### Refactor

- Deprecate old style operation names [CHART-921] (#338)

### Fix

- Use pandas Timedelta types in line and sine_wave functions (#341)
- Return UserRuntimeError in status_flag_filter (#336)
- Change Points Detection chart (#334)
- Add default parameter values to replace/remove [CHART-909] (#335)

### Feat

- Support deprecation of versioned functions [CHART-919] (#337)
- Support for Python 3.9 (#342)
- New function: Holt-Winters forecasting (#316)
- New function: Live fluid properties (#249)

## v2.1.0 (2022-01-05)

### Fix

- Validate inputs for detect algorithms [CHART-855] (#312)
- _make_index Timedelta error and docs build error due to wrong path and type errors (#306)
- Check types for completeness (#293)
- Reference to white_noise in example (#290)
- bin_map returns Series when input is Series [CHART-854] (#284)
- Use Literal type annotations in interpolate and resample functions [CHART-870] (#278)

### Feat

- New function: univariate polynomial (#295)
- Automatically validate input output types (#260)
- New function: sine wave and white noise generation (#268)

## v2.0.0 (2021-12-20)

### Feat

- New function: completeness score (#243)
- Add exceptions types for errors targeted to users [Chart-832] (#252)
- New function: gas density calculator (#231)
- New function: data gap identification algorithms [CHART-759] (#183)
- New function: well status (#224)
- Add wrapper to visualise change points from the ED-Pelt algorithm (#209)
- New function: synthetic signal generator (#62)

### Fix

- Fix path to valve_data.pkl (#259)

## v1.0.0 (2021-12-06)

### Fix

- Fix path to data set (#212)
- Improve error handling in reindex with short time series [CHART-804] (#194)
- Try to handle wrong input types in the steady state detection - cpd (#184)
- Add default parameter values for clip function (#193)

### Feat

- New functions: get_timestamps, get_timestamps and shift [CHART-813] (#172)
- New functions: replace/remove [CHART-789 CHART-808] (#189)
- New function: shut in var (#102)
- Disable extrapolation in auto_align (#163)

## v0.2.1 (2021-11-11)

### Fix

- Complete and fix docstrings for all indsl functions [CHART-739] (#121)
- Solve pandas resampling issue when input datetime index only contains date (#69)

## v0.2.0 (2021-11-04)

### Feat

- Add auto align for inDSL functions [CHART-643] (#61)
- Support for function versioning [CHART-674] (#45)

## v0.1.0 (2021-10-29)

### Fix

- Added missing resample import such that function would be availa… (#65)
- Improved/removed all test warnings #27 (#29)

### Feat

- New function: re-indexing [Chart 626] (#19)

## v0.0.8 (2021-10-18)

### Fix

- Improvements to the ED-Pelt Change Point Detection algorithm (#22)

## v0.0.7 (2021-10-18)

### Fix

- fixes on PR process (#34)
- fix PR process

## v0.0.6 (2021-10-18)

### Fix

- fixes on PR process (#34)

## 0.0.6 (2021-10-18)

### Fix

- fix ci pipeline (#33)

## 0.0.5 (2021-10-18)

### Fix

- add default granularity to integration and differentiation functions (#21)

## v0.0.4 (2021-10-13)

### Fix

- fix github pipelines (#8)
- change name of main branch to master in pipelines (#7)

## 0.0.3 (2021-10-08)
