"""EvalML models."""

try:
    from evalml.automl import AutoMLSearch  # type: ignore
except ImportError as e:
    raise ImportError('EvalML is not installed') from e
import pandas as pd

from ml import BaseEnum, TaskName
from ml.automl import AutoMLEngine
from ml.configuration import Configuration


class EvalMLProblemType(BaseEnum):
    """EvalML problem types."""

    MULTISERIES_TIME_SERIES_REGRESSION = 'multiseries time series regression'
    TIME_SERIES_REGRESSION = 'time series regression'
    TIME_SERIES_BINARY = 'time series binary'
    TIME_SERIES_MULTICLASS = 'time series multiclass'


class EvalMLEngine(AutoMLEngine):
    """EvalML engine for time series modelling."""

    config: Configuration
    automl_algorithms: list[str] = ['default', 'iterative']

    def __init__(self, config: Configuration):
        super().__init__(config)
        if self.config.task == TaskName.CLASSIFICATION:
            raise NotImplementedError('EvalML have not implemented classification yet')
            # self.problem_type = EvalMLProblemType.TIME_SERIES_MULTICLASS
            # self.allowed_model_families = ['multiclass']
        elif self.config.task == TaskName.FORECAST_UNIVARIATE:
            self.problem_type = EvalMLProblemType.TIME_SERIES_REGRESSION
            self.allowed_model_families = ['regression']
        elif self.config.task == TaskName.FORECAST_MULTIVARIATE:
            self.problem_type = EvalMLProblemType.MULTISERIES_TIME_SERIES_REGRESSION
            self.allowed_model_families = ['regression']
        else:
            raise ValueError(
                f'Invalid task "{self.config.task}" for EvalML. Supported tasks: {list(TaskName)}'
            )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        **kwargs,
    ):
        """Perform time series modelling.

        :param pd.DataFrame X_train: Input training features of shape [n_samples, n_features]
        :param pd.Series y_train: target training data of length [n_samples]
        :param pd.DataFrame X_test: Test/val. features of shape [n_samples, n_features]
        :param pd.Series y_test: Test/val. labels of shape [n_samples]
        """
        # Validate inputs
        if isinstance(y_train, pd.DataFrame) and y_train.shape[1] > 1:
            raise NotImplementedError('Multivariate targets not implemented for EvalML')

        # in time series problems, values should be passed in for the time_index, gap,
        # forecast_horizon, and max_delay variables. # For multiseries time series problems, the
        # values passed in should also include the name of a series_id column.
        problem_configuration = {
            'time_index': kwargs['time_index'],
            'gap': kwargs.get('gap', 0),
            'max_delay': kwargs.get('max_delay', 0),
        }
        if self.problem_type in [
            EvalMLProblemType.MULTISERIES_TIME_SERIES_REGRESSION,
            EvalMLProblemType.TIME_SERIES_REGRESSION,
        ]:
            problem_configuration['forecast_horizon'] = kwargs['forecast_horizon']

        # https://evalml.alteryx.com/en/stable/autoapi/evalml/automl/index.html#evalml.automl.AutoMLSearch
        # Defaults:
        # holdout_set_size=0.1
        # max_time, max_iterations, patience, tolerance = all None
        # allow_long_running_models = False
        automl = AutoMLSearch(
            X_train=X_train,
            y_train=y_train,
            X_holdout=X_test,
            y_holdout=y_test,
            random_seed=int(self.config.random_state),
            allowed_model_families=self.allowed_model_families,
            problem_type=self.problem_type,
            automl_algorithm=str(kwargs.get('automl_algorithm', 'default')).lower(),
            problem_configuration=problem_configuration,
            n_jobs=int(self.config.n_jobs),
            verbose=False,
        )
        # Train models
        automl.search()
        self.model = automl.best_pipeline

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Make predictions.

        :param pd.DataFrame X_test: Test/val. features of shape [n_samples, n_features]
        :return pd.Series: Predictions
        """
        raise NotImplementedError('EvalML predict not implemented yet')
