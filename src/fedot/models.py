from fedot.api.main import Fedot
from fedot.core.pipelines.node import PrimaryNode, SecondaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from fedot.utilities.ts_gapfilling import ModelGapFiller
import numpy as np
import pandas as pd

from src.base import Forecaster
from src.util import Utils


class FEDOTForecaster(Forecaster):

    name = 'FEDOT'

    # Training configurations approximately ordered from slowest to fastest
    presets = [
                'fast_train',
                'ts',
                # 'gpu', # Errors with cudf and cuml
                'stable',
                'best_quality', 'auto',
                ]

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95


    def forecast(self, train_df, test_df, forecast_type, horizon, limit, frequency, tmp_dir,
                 nproc=1,
                 preset='fast_train',
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
        :param str preset: Model configuration to use, defaults to 'fast_train'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """

        if forecast_type == 'univariate':
            target_name = 'target'
            train_df.columns = [ target_name ]
            test_df.columns = [ target_name ]

        X_train, y_train, X_test, _ = self.create_tabular_dataset(train_df, test_df, horizon, target_name,
                                                                  tabular_y=False, lag=None)

        # if forecast_type == 'univariate' and 'ISEM_prices' in tmp_dir:
        #     X_test['fedot_datetime'] = X_test.index
        #     X_test['fedot_datetime'] = pd.to_datetime(X_test['fedot_datetime'], errors='coerce')
        #     X_test = X_test[X_test['fedot_datetime'].dt.hour == 0]
        #     X_test = X_test.drop('fedot_datetime', axis=1)
        task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=horizon))

        # Fill in missing gaps in data. Adapted from:
        # https://github.com/ITMO-NSS-team/fedot-examples/blob/main/notebooks/latest/5_ts_specific_cases.ipynb
        def fill_gaps(dataframe):
            # Create model to infer missing values
            node_lagged = PrimaryNode('lagged')
            node_lagged.custom_params = { 'window_size': horizon }
            node_final = SecondaryNode('ridge', nodes_from=[node_lagged])
            pipeline = Pipeline(node_final)
            model_gapfiller = ModelGapFiller(gap_value=-float('inf'), pipeline=pipeline)

            # Filling in the gaps
            data = dataframe.fillna(-float('inf')).copy()
            data = model_gapfiller.forward_filling(data)
            data = model_gapfiller.forward_inverse_filling(data)
            df = pd.DataFrame(data, columns=dataframe.columns)
            return df

        X_train = fill_gaps(X_train)
        X_test = fill_gaps(X_test)

        # Initialize for the time-series forecasting
        model = Fedot(problem='ts_forecasting',
                    task_params=task.task_params,
                    # use_input_preprocessing=True, # fedot>=0.7.0
                    # timeout=limit, # minutes
                    timeout=1, # minutes
                    preset=preset,
                    seed=limit,
                    n_jobs=nproc,
                    )

        model.fit(X_train, y_train)
        model.test_data = X_test

        predictions = self.rolling_origin_forecast(model, X_train, X_test, horizon)
        print('predictions', predictions.shape)
        exit()
        return predictions


    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial limit to use for training models

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Model configuration to use
        :return: Time limit in minutes (int)
        """

        return int((time_limit/60) * self.initial_training_fraction)


    def rolling_origin_forecast(self, model, X_train, X_test, horizon, column=None):
        """Iteratively forecast over increasing dataset

        :param model: Forecasting model, must have predict()
        :param X_train: Training feature data (pandas DataFrame)
        :param X_test: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :param column: Specifies forecast column if dataframe outputted, defaults to None
        :return: Predictions (numpy array)
        """

        # Split test set
        test_splits = Utils.split_test_set(X_test, horizon)

        # Make predictions
        data = X_train
        preds = model.predict(data)
        if column != None:
            preds = preds[column].values

        if len(preds.flatten()) > 0:
            preds = preds[-horizon:]

        predictions = [ preds ]

        # for s in X_test.iterrows():
            # data = pd.concat([data, s])
        for s in test_splits:
            data = pd.concat([data, s])
            data.loc[len(data.index)] = s[1].values

            preds = model.predict(data)
            if column != None:
                preds = preds[column].values

            if len(preds.flatten()) > 0:
                preds = preds[-horizon:]

            predictions.append(preds)

        # print(len(predictions))
        # print(predictions[0].shape)
        # Flatten predictions and truncate if needed
        try:
            predictions = np.concatenate([ p.flatten() for p in predictions ])
        except:
            predictions = np.concatenate([ p.values.flatten() for p in predictions ])
        # print(predictions.shape)
        # predictions = predictions[:len(X_test)]
        # print(predictions.shape)
        return predictions
