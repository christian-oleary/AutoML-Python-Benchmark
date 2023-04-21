import pandas as pd
from autots import AutoTS

from src.abstract import Forecaster


class AutoTSForecaster(Forecaster):

    name = 'AutoTS'


    def forecast(self, train_df, test_df, target_name, horizon, limit, frequency, tmp_dir='./tmp/forecast/autots'):
        """Perform time series forecasting

        :param train_df: Dataframe of training data
        :param test_df: Dataframe of test data
        :param target_name: Name of target variable to forecast (str)
        :param horizon: Forecast horizon (how far ahead to predict) (int)
        :param limit: Iterations limit (int)
        :param frequency: Data frequency (str)
        :param tmp_dir: Path to directory to store temporary files (str)
        """

        model = AutoTS(
            forecast_length=horizon,
            frequency='infer',
            prediction_interval=0.95,
            ensemble=['simple', 'horizontal-min'],
            max_generations=limit,
            num_validations=2,
            validation_method='seasonal 168',
            model_list='superfast',
            transformer_list='all',
            models_to_validate=0.2,
            drop_most_recent=1,
            n_jobs='auto',
        )

        train_df.index = pd.to_datetime(train_df.index)

        model = model.fit(train_df)

        prediction = model.predict()
        forecasts_df = prediction.forecast
        return forecasts_df[target_name]


    def estimate_initial_limit(self, time_limit):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :return: Trials limit (int)
        """

        # return (time_limit / 600) # Estimate a generation takes 10 minutes
        return 1 # One GA generation
