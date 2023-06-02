import os

from fedot.api.main import Fedot
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

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
        :return predictions: TODO
        """

        # model_path = os.path.join(tmp_dir, '0_pipeline_saved', '0_pipeline_saved.json')
        # if not os.path.exists(model_path):
        # Specify the task and the forecast length
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon))

        # Initialize for the time-series forecasting
        model = Fedot(problem='ts_forecasting',
                    task_params=task.task_params,
                    # use_input_preprocessing=True, # fedot>=0.7.0
                    # timeout=limit, # minutes
                    timeout=1, # minutes
                    preset=preset,
                    seed=limit,
                    )

        # Split target from features
        import warnings
        warnings.warn('NOT USING LAGGED FEATURES FROM TARGET VARIABLE')
        y_train = train_df[target_name]
        X_train = train_df.drop(target_name, axis=1)
        X_test = test_df.drop(target_name, axis=1)

        model.fit(X_train, y_train)
        model.test_data = X_test
        #     model.export_as_project(project_path=os.path.join(tmp_dir, 'project.zip'))
        # else:
        #     model.import_as_project(project_path=os.path.join(tmp_dir, 'project.zip'))

        print('y_train', y_train.shape)
        print('X_train', X_train.shape)
        print('X_test', X_test.shape)
        # predictions = model.forecast(X_test)
        # predictions = model.predict(features=X_test)
        predictions = self.rolling_origin_forecast(model, X_train, X_test, horizon)
        # ValueError: all the input arrays must have same number of dimensions, but the array
        # at index 0 has 2 dimension(s) and the array at index 1 has 1 dimension(s)
        print('predictions', predictions, type(predictions))

        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in minutes (int)
        """

        return int((time_limit/60) * self.initial_training_fraction)
