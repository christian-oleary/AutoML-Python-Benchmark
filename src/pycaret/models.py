import numpy as np
import pandas as pd
from pycaret.time_series import TSForecastingExperiment

from src.base import Forecaster
from src.logs import logger
from src.TSForecasting.data_loader import FREQUENCY_MAP


class PyCaretForecaster(Forecaster):

    name = 'PyCaret'

    initial_training_fraction = 0.95 # Use 95% of max. time for trainig in initial experiment

    presets = [ '' ]

    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 target_name=None,
                 preset=''):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param str preset: Model configuration to use, defaults to 'auto'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """

        if forecast_type == 'global':
            freq = FREQUENCY_MAP[frequency].replace('1', '')
            train_df.index = pd.to_datetime(train_df.index).to_period(freq)
            test_df.index = pd.to_datetime(test_df.index).to_period(freq)

        # logger.error(f'horizon {horizon}')
        # # Using the correct horizon causes a variety of internal library errors
        # # during the forecasting stage (via predict_model())
        # # TODO: Use the correct horizon if possible
        # # horizon = len(test_df)
        # logger.error(f'horizon {horizon}')

        exp = TSForecastingExperiment()
        exp.setup(train_df,
                  target=target_name,
                  fh=horizon,
                  fold=2, # Lower folds prevents errors with short time series
                  numeric_imputation_target='ffill',
                  numeric_imputation_exogenous='ffill',
                  use_gpu=True,
                  )

        logger.debug('Training models')
        model = exp.compare_models(budget_time=limit)

        if forecast_type == 'global':
            raise NotImplementedError()
            predictions = exp.predict_model(model, X=test_df.drop(target_name, axis=1), fh=horizon)
            predictions = predictions['y_pred'].values
        else:
            predictions = self.rolling_origin_forecast(exp, model, train_df, test_df, horizon, column='y_pred')
        return predictions


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Time limit in minutes (int)
        """

        return int((time_limit/60) * self.initial_training_fraction)



    def rolling_origin_forecast(self, exp, model, X_train, X_test, horizon, column=None):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (pandas DataFrame)
        :param X_test: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :param column: Specifies forecast column if dataframe outputted, defaults to None
        :return: Predictions (numpy array)
        """

        # Split test set
        from src.util import Utils
        test_splits = Utils.split_test_set(X_test, horizon)

        # Make predictions
        preds = exp.predict_model(model, X=X_train, fh=horizon)
        if column != None:
            preds = preds[column].values
        predictions = [ preds ]

        for s in test_splits:
            X_train = pd.concat([X_train, s])

            preds = exp.predict_model(model, X=X_train, fh=horizon)
            if column != None:
                preds = preds[column].values

            predictions.append(preds)

        # Flatten predictions and truncate if needed
        try:
            predictions = np.concatenate([ p.flatten() for p in predictions ])
        except:
            predictions = np.concatenate([ p.values.flatten() for p in predictions ])
        predictions = predictions[:len(X_test)]
        return predictions

