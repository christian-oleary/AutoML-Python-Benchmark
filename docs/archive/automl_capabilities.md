# AutoML Library Capabilities

Documentation of the functionality of various Python-based AutoML libraries.

- ETNA and [GAMA](https://amore-labs.github.io/website/software/software.html#GAMA) have been discontinued.
- auto-sktime: <https://arxiv.org/abs/2312.08528> - <https://github.com/Ennosigaeon/auto-sktime/> - dead. only supported forecasting.
- TPOT 2 under development: <https://github.com/EpistasisLab/tpot2>
H2O [Driverless AI](https://docs.h2o.ai/driverless-ai/latest-stable/docs/userguide/index.html) is not a standalone library. It is a web pla

## Capability Table

Table of the capabilities of various AutoML libraries based on existing documentation

|                  | Tabular    | Tabular        | Image          |            | Forecasting | Forecasting  | Forecasting  | TSC            | TSC               | TS                | Link
| ---              | ---        | ---            | ---            | ---        | ---         | ---          | ---          | ---            | ---               | ---               | ---
| Library          | Regression | Classification | Classification | Clustering | Univariate  | Multivariate | Global/Panel | By series      | By Point          | Anomaly Detection | Link
| ---              | ---        | ---            | ---            | ---        | ---         | ---          | ---          | ---            | ---               | ---               | ---
| Auto-PyTorch     | Yes        | Yes            | ???            | No         | Yes         | No           | No           | No (1)         | No                | No                | <https://automl.github.io/Auto-PyTorch/master/manual.html>
| Auto-Sklearn     | Yes        | Yes            | ???            | No         | No          | No           | No           | No             | No                | No                | <https://automl.github.io/auto-sklearn/master/manual.html>
| AutoGluon        | Yes        | Yes            | ???            | No         | Yes         | Yes          | Yes          | No             | No                | Yes (2)           | <https://auto.gluon.ai/stable/index.html>
| AutoKeras        | Yes        | Yes            | ???            | No         | No (4)      | No (4)       | No (4)       | No (4)         | No (4)            | Domain-specific   | <https://autokeras.com/>
| AutoTS           | No (3)     | No (3)         | ???            | No         | Yes         | Yes          | No           | No             | No                | Yes               | <https://winedarksea.github.io/AutoTS/build/html/index.html>
| EvalML           | Yes        | Yes            | ???            | No         | Yes         | Yes          | No (5)       | No (6)         | Yes (6)           | No (7)            | <https://evalml.alteryx.com/en/stable/api_index.html#pipelines>
| FEDOT            | Yes        | Yes            | ???            | No         | Yes         | Yes          | No           | No             | No                | No                | <https://fedot.readthedocs.io/en/latest/api/api.html#fedot-api>
| **Fedot.Industrial** | Yes    | Yes            | ???            | No         | Yes         | Yes          | No           | Yes            | No                | No                | <https://github.com/aimclub/Fedot.Industrial>
| FLAML            | Yes        | Yes            | ???            | No         | Yes         | Yes          | Yes          | No  (8)        | Yes (8)           | No                | <https://microsoft.github.io/FLAML/>
| H2O-3            | Yes        | Yes            | ???            | Yes        | No          | No           | No           | No             | No                | No                | <https://docs.h2o.ai/h2o/latest-stable/h2o-docs/index.html>
| **HyperTS**      | No (10)    | No (10)        | No (10)        | No (10)    | Yes         | Yes          | Yes          | Yes            | No (3)            | Yes               | <https://github.com/DataCanvasIO/HyperTS>
| Hyperopt-sklearn | Yes        | Yes            | ???            | Yes        | No          | No           | No           | No (9)         | No                | No                | <https://github.com/hyperopt/hyperopt-sklearn>
| KATS             | No         | No             | ???            | No         | Yes         | Yes          | Yes          | No             | No                | Yes               | <https://facebookresearch.github.io/Kats/>
| LightAutoML      | Yes        | Yes            | ???            | No         | No          | No           | No           | No             | No                | No                | <https://lightautoml.readthedocs.io/en/v.0.4.0/pages/tutorials/Tutorial_1_basics.html#1.1.-Task-type>
| Ludwig           | Yes        | Yes            | ???            | No         | Yes         | Yes          | Yes          | No             | No                | No                | <https://github.com/ludwig-ai/ludwig>
| MLBox            | Yes        | Yes            | ???            | No         | No          | No           | No           | No             | No                | No                | <https://mlbox.readthedocs.io/en/latest/index.html>
| mljar-supervised | Yes        | Yes            | ???            | No         | No          | No           | No           | No             | No                | No                | <https://supervised.mljar.com/api/>
| PyCaret          | Yes        | Yes            | ???            | Yes        | Yes         | Yes          | No           | No             | No                | Yes (2)           | <https://pycaret.readthedocs.io/en/latest/tutorials.html>
| TPOT             | Yes        | Yes            | ???            | No         | No          | No           | No           | No             | No                | No                | <https://github.com/EpistasisLab/tpot>

| NAMENAME         | ???        | ???            | ???            | ???        | ???         | ???          | ???          | ???            | ???               | ???               | <>

- Notes:
    1. Auto-PyTorch does [not support TS classification](https://github.com/automl/Auto-PyTorch/issues/491).
    2. Yes, but functionality is not time series-specific
        - AutoGluon has some basic non-TS-related anomaly detection in its [EDA module](https://github.com/autogluon/autogluon/blob/master/eda/src/autogluon/eda/analysis/anomaly.py).
    3. Not intended behaviour
    4. AutoKeras time series functionality [removed](https://github.com/keras-team/autokeras/issues/1859) by [version 2.0.0](https://github.com/keras-team/autokeras/blob/db78b445ee3aa3aedb19a71d2d1e330cc87f12b3/RELEASE.md?plain=1#L16)
    5. EvalML can forecast for multiple time series simultaneously but [assumes they are independent](https://evalml.alteryx.com/en/stable/user_guide/timeseries.html#What-is-multiseries?).
    6. Point classification only. EvalML claims to have time series classification but documentation is non-existant and source code indicates that this functionality is not fully implemented yet.
    7. EvalML listed here as having no functionality for anomaly detection but its data validation module does have a function to apply [IQR](https://evalml.alteryx.com/en/stable/autoapi/evalml/data_checks/index.html#evalml.data_checks.OutliersDataCheck)
    8. Point classification only ([FLAML](https://microsoft.github.io/FLAML/docs/Examples/AutoML-Time%20series%20forecast/#forecasting-discrete-variables))
    9. KATS does not have time series classification classes or functions but it does have TS feature extraction: <https://github.com/facebookresearch/Kats/blob/main/tutorials/kats_203_tsfeatures.ipynb>
    10. HyperTS is TS only but the maintainers have other projects: Hypernets, HyperGBM, HyperDT/DeepTables, HyperKeras

## Definitions

- **EvalML**:
  - ***Time Series***: "A time series is a series of measurements taken at different moments in time (Wikipedia). The main difference between a time series dataset and a normal dataset is that the rows of a time series dataset are ordered chronologically, where the relative time between rows is significant." - <https://evalml.alteryx.com/en/stable/user_guide/timeseries.html#But-first,-what-is-a-time-series?>
  - ***Multiseries time series***: Multiseries time series refers to data where we have multiple time series that we’re trying to forecast simultaneously. For example, if we are a retailer who sells multiple products at our stores, we may have a single dataset that contains sales data for those multiple products. In this case, we would like to forecast sales for all products without splitting them off into separate datasets.
  There are two forms of multiseries forecasting - independent and dependent. Independent forecasting assumes that the separate series we’re modeling are independent from each other, that is, the value of one series at a given point in time is unrelated to the value of a different series at any point in time. In our sales example, product A sales and product B sales would not impact each other at all. Dependent forecasting is the opposite, where it is assumed that all series have an impact on the others in the dataset. At the moment, ***EvalML only supports independent multiseries time series forecasting***." - <https://evalml.alteryx.com/en/stable/user_guide/timeseries.html#What-is-multiseries?>
