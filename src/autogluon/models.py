# from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from src.abstract import Forecaster


class AutoGluonForecaster(Forecaster):

    name = 'AutoGluon'

    def forecast(self):
        raise NotImplementedError()
