import numpy as np
import pandas as pd
from autots import AutoTS

from src.base import Forecaster
from src.errors import DatasetTooSmallError
from src.logs import logger
from src.TSForecasting.data_loader import FREQUENCY_MAP
from src.util import Utils


class AutoTSForecaster(Forecaster):

    name = 'AutoTS'

    # Training configurations ordered from slowest to fastest
    presets = [ 'all', 'default', 'fast_parallel', 'fast', 'superfast' ]

    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 preset='superfast',
                 target_name=None):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param str preset: Model configuration to use, defaults to 'all'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """

        # if forecast_type == 'global':
        #     freq = FREQUENCY_MAP[frequency].replace('1', '').replace('min', 'T')
        # else:
        #     freq = frequency

        train_df.index = pd.to_datetime(train_df.index)
        test_df.index = pd.to_datetime(test_df.index)

        # from sklearn.impute import IterativeImputer
        # imputer = IterativeImputer(n_nearest_features=3, max_iter=5)
        # train_df = pd.DataFrame(imputer.fit_transform(train_df), index=train_df.index, columns=train_df.columns)
        # test_df = pd.DataFrame(imputer.transform(test_df), index=test_df.index, columns=test_df.columns)

        min_allowed_train_percent = 0.1
        model = AutoTS(
            ensemble=['auto'],
            # frequency=freq,
            frequency='infer',
            forecast_length=horizon,
            max_generations=limit,
            model_list=preset,
            models_to_validate=0.2,
            n_jobs=1,
            # num_validations=1,
            prediction_interval=0.95,
            random_seed=limit,
            transformer_list='all',
            # validation_method='similarity',
            min_allowed_train_percent=min_allowed_train_percent
        )

        logger.debug('Training model...')

        print('horizon', horizon)
        print('min_allowed_train_percent', min_allowed_train_percent)
        print('train_df.shape[0]', train_df.shape[0])
        print('horizon * min_allowed_train_percent', horizon * min_allowed_train_percent)
        print('int((train_df.shape[0]) - horizon)', int((train_df.shape[0]) - horizon))
        print((horizon * min_allowed_train_percent) > int(
            (train_df.shape[0]) - horizon))

        # horizon 4
        # min_allowed_train_percent 0.5
        # train_df.shape[0] 16
        # horizon * min_allowed_train_percent 2.0
        # int((train_df.shape[0]) - horizon) 12
        # False

        # forecast_length 4
        # min_allowed_train_percent 0.5
        # df.shape[0] 4
        # forecast_length * min_allowed_train_percent 2.0
        # int((df.shape[0]) - forecast_length) 0
        # True

        if (horizon * min_allowed_train_percent) > int((train_df.shape[0]/4) - horizon):
            raise DatasetTooSmallError('Time series is too short for AutoTS', ValueError())

        # We need to pass future_regressor to be able to do rolling origin forecasting
        model = model.fit(train_df)#, future_regressor=train_df)
        # model = model.fit(train_df, future_regressor=train_df)
        exit()

        logger.debug('Making predictions...')
        predictions = self.rolling_origin_forecast(model, train_df, test_df, horizon)
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
