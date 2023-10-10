import os
import logging

import pandas as pd
from ludwig.api import LudwigModel
from ludwig.utils.data_utils import add_sequence_feature_column

from src.base import Forecaster


class LudwigForecaster(Forecaster):

    name = 'Ludwig'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment

    presets = [ 10, 100, 1000 ]

    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 nproc=1,
                 preset=10,
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
        :param int preset: Number of epochs, defaults to 10
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """

        # backfill, forwardfill

        if forecast_type == 'univariate':
            target_name = 'target'
            train_df.columns = [target_name]
            test_df.columns = [target_name]

        # IGNORE. Produces scaled predictions...
        # Ludwig examples indicate scaling must be done separately: https://ludwig.ai/latest/examples/weather/
        # train_df[target_name] = ((train_df[target_name]-train_df[target_name].mean()) / train_df[target_name].std())
        # test_df[target_name] = ((test_df[target_name]-test_df[target_name].mean()) / test_df[target_name].std())

        if forecast_type == 'univariate':
            target_name = 'target'
            train_df.columns = [target_name]
            test_df.columns = [target_name]

        # Format DataFrame
        add_sequence_feature_column(train_df, target_name, self.get_default_lag(horizon))
        add_sequence_feature_column(test_df, target_name, self.get_default_lag(horizon))


        config = {
            'input_features': [{'name': f'{target_name}_feature', 'type': 'timeseries'}],
            'output_features': [{ 'name': target_name, 'type': 'numerical' }],
            'trainer': { 'epochs': int(preset) }
        }

        # Constructs Ludwig model from config dictionary
        model = LudwigModel(config, logging_level=logging.WARNING)

        model.train(dataset=train_df, output_directory=tmp_dir,
                    skip_save_training_description=True,
                    skip_save_training_statistics=True,
                    skip_save_model=True,
                    skip_save_progress=True,
                    skip_save_log=True,
                    skip_save_processed_input=True,
                    )

        _, predictions, __ = model.evaluate(test_df, collect_predictions=True, collect_overall_stats=True,
                                            output_directory=os.path.join(tmp_dir, 'evaluate'))

        predictions = predictions[f'{target_name}_predictions'].values
        return predictions


    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Model configuration to use
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)
