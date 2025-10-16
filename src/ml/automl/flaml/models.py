"""FLAML models."""

try:
    from flaml import AutoML
except ModuleNotFoundError:
    raise ModuleNotFoundError('FLAML not installed')
from loguru import logger
import pandas as pd

from ml import TaskName
from ml.automl import AutoMLEngine
from ml.configuration import Configuration
from ml.errors import DatasetTooSmallError


class FLAMLEngine(AutoMLEngine):
    """FLAMLEngine engine for time series modelling."""

    config: Configuration

    def __init__(self, config: Configuration):
        super().__init__(config)
        # 'classification', 'regression', 'ts_forecast''
        if self.config.task == TaskName.CLASSIFICATION:
            self.task = 'ts_forecast_classification'
        elif self.config.task == TaskName.FORECAST_UNIVARIATE:
            self.task = 'ts_forecast'
        elif self.config.task == TaskName.FORECAST_MULTIVARIATE:
            self.task = 'ts_forecast_panel'
        else:
            raise ValueError(f'Invalid task "{self.config.task}" for FLAML')

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
        :param pd.DataFrame X_test: Test or val. features of shape [n_samples, n_features]
        :param pd.Series y_test: Unused. Test or val. labels of shape [n_samples]
        """
        if kwargs.get('horizon') and len(X_test) <= kwargs['horizon'] + 1:  # 4 = lags
            raise DatasetTooSmallError('Dataset too small for FLAML', ValueError())

        extra_kwargs = {}
        if 'forecast' in self.task:
            extra_kwargs['period'] = kwargs['horizon']

        # "For time series forecast tasks, the first column of X_train must be
        # the timestamp column (datetime type). Other columns in the dataframe
        # are assumed to be exogenous variables (categorical or numeric)."

        automl = AutoML()
        logger.debug('Training models via FLAML...')
        automl.fit(
            X_train=X_train,  # train_df.index.to_series(name='ds').values,
            y_train=y_train,
            estimator_list='auto',
            eval_method='auto',
            n_jobs=self.config.n_jobs,
            task=self.task,
            verbose=0,  # Higher = more messages
            # time_budget=limit,  # seconds
            # log_file_name=os.path.join(tmp_dir, self.task, f'{self.task}.log'),
        )
        logger.debug('Training with FLAML finished.')
        # predictions = automl.predict(test_df.index.to_series(name='ds').values, period=horizon).values
        # return predictions

    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Make predictions.

        :param pd.DataFrame X_test: Test or validation features of shape [n_samples, n_features]
        :return pd.Series: Predictions
        """
        raise NotImplementedError('FLAML predict not implemented yet')
