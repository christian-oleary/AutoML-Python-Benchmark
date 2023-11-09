import itertools
import os
import shutil

import autokeras as ak

from src.base import Forecaster
from src.errors import AutomlLibraryError

# Presets are every combination of the following:
optimizers = ['greedy', 'bayesian', 'hyperband', 'random']
epoch_limits = ['10', '50', '100', '150']
time_limits = ['60', '300', '600'] # 1 min, 5 min, 10 min
presets = list(itertools.product(optimizers, epoch_limits, time_limits))
presets = [ '_'.join(p) for p in presets ]

class AutoKerasForecaster(Forecaster):

    name = 'AutoKeras'

    # Training configurations (not ordered)
    presets = presets


    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 nproc=1,
                 preset='greedy_32_60',
                 target_name=None):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param int nproc: Number of threads/processes allowed, defaults to 1
        :param str preset: Model configuration to use
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """

        self.forecast_type = forecast_type

        # Cannot use tmp_dir due to internal bugs with AutoKeras
        tmp_dir = 'time_series_forecaster'
        shutil.rmtree(tmp_dir, ignore_errors=True)

        if forecast_type == 'univariate':
            target_name = 'target'
            train_df.columns = [ target_name ]
            test_df.columns = [ target_name ]
            lag = 1 # AK has lookback
            X_train, y_train, X_test, _ = self.create_tabular_dataset(train_df, test_df, horizon, target_name,
                                                                      tabular_y=False, lag=lag)
        else:
            raise NotImplementedError()

        optimizer = preset.split('_')[0]
        epochs = int(preset.split('_')[1])
        tmp_dir = os.path.join(tmp_dir, f'{optimizer}_{epochs}epochs_{limit}')

        # Initialise forecaster
        params = {
            # 'directory': tmp_dir, # Internal errors with AutoKeras
            'lookback': self.get_default_lag(horizon),
            'max_trials': int(limit),
            'objective': 'val_loss',
            'overwrite': False,
            'predict_from': 1,
            'predict_until': horizon,
            'seed': int(limit),
            'tuner': optimizer,
        }
        clf = ak.TimeseriesForecaster(**params)

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
            x=X_train,
            y=y_train,
            # validation_split=0.2, # Internal errors
            validation_data=(X_train, y_train),
            batch_size=batch_size,
            epochs=epochs,
            verbose=0
        )

        predictions = self.rolling_origin_forecast(clf, X_train, X_test, horizon)
        if len(predictions) == 0:
            raise AutomlLibraryError('AutoKeras failed to produce predictions', ValueError())
        return predictions


    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Model configuration to use
        :return: Trials limit (int)
        """
        return int(time_limit / int(preset.split('_')[2]))
