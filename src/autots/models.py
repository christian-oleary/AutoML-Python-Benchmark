import numpy as np
import pandas as pd
from autots import AutoTS

from src.abstract import Forecaster
from src.util import Utils
from src.TSForecasting.data_loader import FREQUENCY_MAP


class AutoTSForecaster(Forecaster):

    name = 'AutoTS'

    # Training configurations ordered from slowest to fastest
    presets = [ 'all', 'default', 'fast_parallel', 'fast', 'superfast' ]

    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir, preset='all'):
        """Perform time series forecasting

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Iterations limit (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        :param preset: Which modelling option option to use (str)
        :return predictions: TODO
        """

        freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')

        train_df = train_df[target_name]
        test_df = test_df[target_name]

        train_df.index = pd.to_datetime(train_df.index)
        test_df.index = pd.to_datetime(test_df.index)

        model = AutoTS(
            ensemble=['auto'],
            frequency=freq,
            # frequency='infer',
            forecast_length=horizon,
            max_generations=limit,
            model_list=preset,
            models_to_validate=0.2,
            n_jobs='auto',
            # n_jobs=1,
            # num_validations=2,
            prediction_interval=0.95,
            random_seed=limit,
            transformer_list='all',
            validation_method='similarity',
        )

        # We need to pass future_regressor to be able to do rolling origin forecasting
        model = model.fit(train_df, future_regressor=train_df)
        predictions = self.rolling_origin_forecast(model, train_df, test_df, horizon)

        # model = model.fit(train_df)
        # predictions = model.predict(len(test_df), just_point_forecast=True)[target_name].values
        # predictions = prediction.forecast[target_name].values

        # print(model)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Trials limit (int)
        """

        # return (time_limit / 600) # Estimate a generation takes 10 minutes
        return 1 # One GA generation


    def rolling_origin_forecast(self, model, train_X, test_X, horizon):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param train_X: Training feature data (pandas DataFrame)
        :param test_X: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :return: Predictions (numpy array)
        """
        # Split test set
        test_splits = Utils.split_test_set(test_X, horizon)

        # Make predictions
        # preds = model.predict(future_regressor=train_X, forecast_length=horizon).forecast.values
        predictions = []
        for s in test_splits:
            # train_X = pd.concat([train_X, s])
            preds = model.predict(future_regressor=s, forecast_length=horizon).forecast.values
            predictions.append(preds)

        # Flatten predictions and truncate if needed
        predictions = np.concatenate([ p.flatten() for p in predictions ])
        predictions = predictions[:len(test_X)]
        return predictions
