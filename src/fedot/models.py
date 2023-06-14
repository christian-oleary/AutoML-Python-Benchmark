import os

from fedot.api.main import Fedot
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.utilities.ts_gapfilling import ModelGapFiller
import pandas as pd

from src.abstract import Forecaster


class FEDOTForecaster(Forecaster):

    name = 'FEDOT'

    # Training configurations approximately ordered from slowest to fastest
    presets = [ 'best_quality', 'auto', 'gpu', 'stable', 'ts', 'fast_train' ]

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir, preset='fast_train'):
        """Perform time series forecasting

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Iterations limit (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        :param preset: Model configuration to use
        :return predictions: TODO
        """

        # Split target from features
        y_train = train_df[target_name]
        X_train = train_df.drop(target_name, axis=1)
        X_test = test_df.drop(target_name, axis=1)

        # model_path = os.path.join(tmp_dir, '0_pipeline_saved', '0_pipeline_saved.json')
        # if not os.path.exists(model_path):
        # Specify the task and the forecast length
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
        #     model.export_as_project(project_path=os.path.join(tmp_dir, 'project.zip'))
        # else:
        #     model.import_as_project(project_path=os.path.join(tmp_dir, 'project.zip'))

        predictions = self.rolling_origin_forecast(model, X_train, X_test, horizon)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in minutes (int)
        """

        return int((time_limit/60) * self.initial_training_fraction)
