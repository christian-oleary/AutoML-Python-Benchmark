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

        # IGNORE. Produces scaled predictions...
        # Ludwig examples indicate scaling must be done separately: https://ludwig.ai/latest/examples/weather/
        # train_df[target_name] = ((train_df[target_name]-train_df[target_name].mean()) / train_df[target_name].std())
        # test_df[target_name] = ((test_df[target_name]-test_df[target_name].mean()) / test_df[target_name].std())

        if forecast_type == 'univariate':
            target_name = 'target'
            train_df.columns = [target_name]
            test_df.columns = [target_name]

            # Generate target variables
            self.create_future_values(train_df, horizon, target_name)
            self.create_future_values(test_df, horizon, target_name)

            # Generate features
            train_features = train_df[[target_name]]
            add_sequence_feature_column(train_features, target_name, self.get_default_lag(horizon))
            train_df = pd.concat([train_df.drop(target_name, axis=1), train_features], axis=1)

            test_features = test_df[[target_name]]
            test_features = pd.concat([train_features.tail(horizon), test_features], axis=0) # add training data for lags calculation
            add_sequence_feature_column(test_features, target_name, self.get_default_lag(horizon))
            test_features = test_features.tail(len(test_features) - horizon) # drop training data after lags calculation
            test_df = pd.concat([test_df.drop(target_name, axis=1), test_features], axis=1)
        else:
            raise NotImplementedError()

        config = {
            'input_features': [{'name': f'{target_name}_feature', 'type': 'timeseries'}],
            'output_features': [{ 'name': target_name, 'type': 'numerical' }] + [
                { 'name': f'{target_name}+{i}', 'type': 'numerical' }
                for i in range(1, 24)
            ],
            'trainer': { 'epochs': int(preset) }
        }

        # Drop irrelevant rows
        if forecast_type == 'univariate' and 'ISEM_prices' in tmp_dir:
            test_df['ludwig_datetime'] = test_df.index
            test_df['ludwig_datetime'] = pd.to_datetime(test_df['ludwig_datetime'], errors='coerce')
            test_df = test_df[test_df['ludwig_datetime'].dt.hour == 0]
            test_df = test_df.drop('ludwig_datetime', axis=1)

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

        predictions, _ = model.predict(test_df)
        predictions = predictions.values.flatten()

        # Results unrealistically optimistic
        # _, predictions, __ = model.evaluate(test_df, collect_predictions=True, collect_overall_stats=True,
        #                                     output_directory=os.path.join(tmp_dir, 'evaluate'))

        # class Model:
        #     def __init__(self, ludwig_model):
        #         self.ludwig_model = ludwig_model

        #     def predict(self, df):
        #         predictions, _ = self.ludwig_model.predict(df)
        #         return predictions
        # predictions = self.rolling_origin_forecast(Model(model), train_df, test_df, horizon,
        #                                            column=f'{target_name}_predictions')
        return predictions


    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Model configuration to use
        :return: Time limit in seconds (int)
        """

        return int(time_limit * self.initial_training_fraction)


    def rolling_origin_forecast(self, model, X_train, X_test, horizon, column=None):
        """DEPRECATED Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (pandas DataFrame)
        :param X_test: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :param column: Specifies forecast column if dataframe outputted, defaults to None
        :return: Predictions (numpy array)
        """

        # # Split test set
        # from util import Utils
        # test_splits = Utils.split_test_set(X_test, horizon)

        # # Make predictions
        # data = X_train[-horizon:]
        # print('data.shape', data.shape)
        # print(data)
        # print(data[0])
        # exit()
        # preds = model.predict(data[0])
        # if column != None:
        #     preds = preds[column].values

        # print('0 preds', preds.shape, len(preds), len(preds) > horizon)
        # if len(preds) > horizon:
        #     preds = preds[-horizon:]
        # print('0 preds.shape', preds.shape)

        # predictions = [ preds ]

        # for s in test_splits:
        #     data = pd.concat([data, s])

        #     preds = model.predict(data)
        #     print('preds.shape', preds.shape)
        #     if column != None:
        #         preds = preds[column].values

        #     predictions.append(preds)

        # # Flatten predictions and truncate if needed
        # import numpy as np
        # try:
        #     predictions = np.concatenate([ p.flatten() for p in predictions ])
        # except:
        #     predictions = np.concatenate([ p.values.flatten() for p in predictions ])
        # print('predictions.shape', predictions.shape)
        # predictions = predictions[:len(X_test)]
        # print('predictions.shape', predictions.shape)
        # return predictions