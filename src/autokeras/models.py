import os
import shutil

import autokeras as ak
import pandas as pd
from sklearn.impute import IterativeImputer
import tensorflow as tf

from src.abstract import Forecaster


class AutoKerasForecaster(Forecaster):

    name = 'AutoKeras'

    # Training configurations (not ordered)
    presets = ['greedy', 'bayesian', 'hyperband', 'random']


    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 preset='greedy',
                 **kwargs):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param preset: Model configuration to use
        :return predictions: Numpy array of predictions
        """

        # Cannot use tmp_dir due to internal bugs with AutoKeras
        tmp_dir = 'time_series_forecaster'
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if forecast_type == 'univariate':
            step_size = kwargs['step_size']
            target_name = 'target'
            train_df.columns = [ target_name ]
            test_df.columns = [ target_name ]

            train_y = train_df[target_name]

            # Provide AutoKeras with some feature data
            train_X = train_df[[target_name]].shift(step_size)
            test_X = test_df[[target_name]].shift(step_size)
            imputer = IterativeImputer(max_iter=5, random_state=0)
            train_X = pd.DataFrame(imputer.fit_transform(train_X), columns=[target_name])
            test_X = pd.DataFrame(imputer.fit_transform(test_X), columns=[target_name])

            objective = 'val_loss'

        else:
            import warnings
            warnings.warn('NOT USING LAGGED FEATURES FROM TARGET VARIABLE')

            # Split target from features
            target_name = kwargs['target_name']
            train_y = train_df[target_name]
            train_X = train_df.drop(target_name, axis=1)
            test_X = test_df.drop(target_name, axis=1)

            objective = 'val_loss'

        epochs = 1000 # AK default
        tmp_dir = os.path.join(tmp_dir, f'{preset}_{epochs}epochs')

        # Initialise forecaster
        params = {
            # 'directory': tmp_dir, # Internal errors with AutoKeras
            'lookback': horizon,
            'max_trials': limit,
            'objective': objective,
            'overwrite': False,
            'predict_from': 1,
            'predict_until': horizon,
            'seed': limit,
            'tuner': preset,
        }
        clf = ak.TimeseriesForecaster(**params)

        # model_path = os.path.join(tmp_dir, 'time_series_forecaster', 'best_pipeline')
        # print(tmp_dir)
        # print(model_path)
        # if not os.path.exists(model_path):

        # "lookback" must be divisable by batch size due to library bug:
        # https://github.com/keras-team/autokeras/issues/1720
        # Start at 512 as batch size and decrease until a factor is found
        # Counting down prevents unnecessarily small batch sizes being selected
        batch_size = None
        size = 512 # Prospective batch size
        while batch_size == None:
            if (horizon / size).is_integer(): # i.e. is a factor
                batch_size = size
            else:
                size -= 1

        # Train models
        clf.fit(
            x=train_X,
            y=train_y,
            validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0
        )

        try:  # Issue with AutoKeras and tmp dir
            predictions = self.rolling_origin_forecast(clf, train_X, test_X, horizon, **kwargs)
            # print(clf.tuner.best_pipeline_path) # AK bug: This is wrong if "directory" is set
            assert len(predictions) > 0
        except AssertionError:
            clf = ak.TimeseriesForecaster(params)
            predictions = self.rolling_origin_forecast(clf, train_X, test_X, horizon, **kwargs)
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Trials limit (int)
        """

        # return int(time_limit / 900) # Estimate a trial takes about 15 minutes
        return 1 # One trial
