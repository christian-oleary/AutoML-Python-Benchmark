import os
import logging

import pandas as pd
from ludwig.api import LudwigModel

from src.abstract import Forecaster


class LudwigForecaster(Forecaster):

    name = 'Ludwig'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir):
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

        config = {
            'input_features': [
                {
                    'name': feature_name,
                    'type': 'number',
                    'preprocessing': {'num_processes': 1}, # TODO
                }
                for feature_name in train_df.columns if feature_name != target_name
            ],
            'output_features': [{ 'name': target_name, 'type': 'number'} ],
            'trainer': { 'epochs': 5 } # TODO: limit?
        }

        # Constructs Ludwig model from config dictionary
        model = LudwigModel(config, logging_level=logging.INFO)

        train_stats, preprocessed_data, output_directory = model.train(dataset=train_df, output_directory=tmp_dir)
        print('train_stats', train_stats)
        print('preprocessed_data', preprocessed_data)
        print('output_directory', output_directory)

        test_stats, predictions, output_directory = model.evaluate(
            test_df,
            collect_predictions=True,
            collect_overall_stats=True
        )
        print('test_stats', test_stats)
        print('output_directory', output_directory)

        predictions = predictions[f'{target_name}_predictions']

        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)
