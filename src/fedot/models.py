import os

from fedot.api.main import Fedot
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.utilities.ts_gapfilling import ModelGapFiller
import pandas as pd

from src.base import Forecaster


class FEDOTForecaster(Forecaster):

    name = 'FEDOT'

    # Training configurations approximately ordered from slowest to fastest
    presets = [ 'best_quality', 'auto', 'gpu', 'stable', 'ts', 'fast_train' ]

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95


    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 preset='fast_train',
                 target_name=None):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param str preset: Model configuration to use, defaults to 'default'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """

        X_train, y_train, X_test = self.create_tabular_dataset(train_df, test_df, horizon, target_name,
                                                                   tabular_y=False, lag=None)

        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon))

        # Fill in missing gaps in data. Adapted from:
        # https://github.com/ITMO-NSS-team/fedot-examples/blob/main/notebooks/latest/5_ts_specific_cases.ipynb
        def fill_gaps(dataframe):
            # Create model to infer missing values
            node_lagged = PrimaryNode('lagged')
            node_lagged.custom_params = { 'window_size': horizon }
            node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
            pipeline = Pipeline(node_final)
            model_gapfiller = ModelGapFiller(gap_value=-float('inf'), pipeline=pipeline)

            # Filling in the gaps
            data = dataframe.fillna(-float('inf')).copy()
            data = model_gapfiller.forward_filling(data)
            data = model_gapfiller.forward_inverse_filling(data)
            df = pd.DataFrame(data, columns=dataframe.columns)
            return df

        X_train = fill_gaps(X_train)
        X_test = fill_gaps(X_test)

        # Initialize for the time-series forecasting
        model = Fedot(problem='ts_forecasting',
                    task_params=task.task_params,
                    # use_input_preprocessing=True, # fedot>=0.7.0
                    timeout=limit, # minutes
                    preset=preset,
                    seed=limit,
                    # n_jobs=-1,
                    )

        model.fit(X_train, y_train)
        model.test_data = X_test

        predictions = self.rolling_origin_forecast(model, X_train, X_test, horizon)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in minutes (int)
        """

        return int((time_limit/60) * self.initial_training_fraction)
