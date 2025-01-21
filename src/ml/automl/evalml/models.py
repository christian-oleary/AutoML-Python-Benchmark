"""EvalML models."""

try:
    from evalml.automl import AutoMLSearch  # type: ignore
except ImportError as e:
    raise ImportError('EvalML is not installed') from e
import pandas as pd

from ml.base import Forecaster
from ml.errors import DatasetTooSmallError

try:
    from evalml.automl import AutoMLSearch  # type: ignore
except ModuleNotFoundError as error:
    raise ModuleNotFoundError('EvalML not installed') from error


class EvalMLForecaster(Forecaster):
    """EvalML forecaster."""

    name = 'EvalML'

    # Training configurations ordered from slowest to fastest
    # presets = [ 'default', 'iterative' ]
    presets = ['default']

    # Use 95% of maximum available time for model training in initial experiment
    initial_training_fraction = 0.95

    def forecast(
        self,
        train_df,
        test_df,
        forecast_type,
        horizon,
        limit,
        frequency,
        tmp_dir,
        nproc=1,
        preset='default',
        target_name=None,
    ):
        """Perform time series forecasting

        :param pd.DataFrame train_df: Dataframe of training data
        :param pd.DataFrame test_df: Dataframe of test data
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :param int horizon: Forecast horizon (how far ahead to predict)
        :param int limit: Time limit in seconds
        :param int frequency: Data frequency
        :param str tmp_dir: Path to directory to store temporary files
        :param int nproc: Number of threads/processes allowed, defaults to 1
        :param str preset: Model configuration to use, defaults to 'default'
        :param str target_name: Name of target variable for multivariate forecasting, defaults to None
        :return predictions: Numpy array of predictions
        """
        raise NotImplementedError()

        # Prepare data
        if target_name is None:
            target_name = 'target'
            train_df.columns = [target_name]
            test_df.columns = [target_name]

        X_train, y_train, X_test, _ = self.create_tabular_dataset(
            train_df, test_df, horizon, target_name, tabular_y=False
        )

        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        if forecast_type == 'global':
            raise NotImplementedError()

        if 'ISEM_prices' in tmp_dir:
            X_train['time_index'] = pd.to_datetime(X_train.index, format='%d/%m/%Y %H:%M')
            X_test['time_index'] = pd.to_datetime(X_test.index, format='%d/%m/%Y %H:%M')
            y_train = pd.Series(y_train, index=X_train.index)
        else:
            freq = 'D'
            X_train['time_index'] = pd.to_datetime(X_train.index, unit=freq)
            X_test['time_index'] = pd.to_datetime(X_test.index, unit=freq)
            y_train = pd.Series(y_train)

        problem_config = {
            'gap': 0,
            'max_delay': horizon,  # for feature engineering
            'forecast_horizon': horizon,
            'time_index': 'time_index',
        }

        eval_size = horizon * 3  # as n_splits=3
        train_size = len(train_df) - eval_size
        window_size = problem_config['gap'] + problem_config['max_delay'] + horizon
        if train_size <= window_size:
            raise DatasetTooSmallError('Time series is too short for EvalML. Must be > 5*horizon')

        automl = AutoMLSearch(
            X_train,
            y_train,
            allowed_model_families=['regression'],
            automl_algorithm=preset,
            problem_type='time series regression',
            problem_configuration=problem_config,
            max_time=limit,
            n_jobs=nproc,
            verbose=False,
        )

        automl.search()
        model = automl.best_pipeline
        predictions = self.rolling_origin_forecast(
            model, X_train, X_test, y_train, horizon, forecast_type, tmp_dir
        )
        return predictions

    def estimate_initial_limit(self, time_limit, preset):
        """Estimate initial limit to use for training models.

        :param time_limit: Maximum amount of time allowed for forecast() (int)
        :param str preset: Model configuration to use
        :return: Time limit in seconds (int)
        """
        return int(time_limit * self.initial_training_fraction)

    def rolling_origin_forecast(
        self, model, train_X, test_X, y_train, horizon, forecast_type, tmp_dir
    ):
        """Iteratively forecast over increasing dataset.

        :param model: Forecasting model, must have predict()
        :param train_X: Training feature data (pandas DataFrame)
        :param test_X: Test feature data (pandas DataFrame)
        :param horizon: Forecast horizon (int)
        :param column: Specifies forecast column if dataframe outputted, defaults to None
        :param str forecast_type: Type of forecasting, i.e. 'global', 'multivariate' or 'univariate'
        :return: Predictions (numpy array)
        """
        raise NotImplementedError()
        # # Split test set
        # test_splits = Utils.split_test_set(test_X, horizon)
        # logger.debug('len(test_splits)', len(test_splits))

        # predictions = []
        # # for s in test_X.iterrows():
        # #     logger.debug('s')
        # #     logger.debug(s, type(s))
        # #     logger.debug(s[1], type(s[1]))
        # #     exit()
        # #     s = s[1].values.flatten()

        # # for i in range(len(test_X)):
        # #     logger.debug(i)
        # #     s = test_X.iloc[[i]]
        # #     train_X = pd.concat([train_X, s])

        # test = []
        # for s in test_splits:
        #     if forecast_type == 'univariate' and 'ISEM_prices' in tmp_dir:
        #         s = s.head(1)
        #         s.index = pd.to_datetime(s['time_index'], format='%d/%m/%Y %H:%M')
        #         s = s.values.flatten()[-horizon:]
        #         test.append(s)
        # predictions = model.predict(test, objective=None, X_train=train_X, y_train=y_train).values
        # logger.debug('predictions.shape', predictions.shape)
        # exit()

        # #     # if horizon > len(s): # Pad with zeros to prevent errors with ARIMA
        # #     #     logger.debug('A')
        # #     #     padding = horizon - len(s)
        # #     #     s = pd.concat([s, pd.DataFrame([s.values[0].tolist()] * padding, columns=s.columns)])
        # #     #     start_index = s.index.values[0]
        # #     #     try:
        # #     #         s.index = np.arange(start_index, start_index + len(s))
        # #     #         if forecast_type == 'univariate':
        # #     #             s['time_index'] = pd.date_range(start=s['time_index'].values[0], periods=len(s))
        # #     #     except: # Datetime indices
        # #     #         if forecast_type == 'univariate':
        # #     #             s['time_index'] = pd.date_range(start=s['time_index'].values[0], periods=len(s))
        # #     #     preds = model.predict(s, objective=None, X_train=train_X, y_train=y_train).values
        # #     #     preds = preds[:len(s)] # Drop placeholder predictions
        # #     # else:
        # #     logger.debug('B')
        # #     # try:
        # #     # logger.debug('train_X.tail(1).index', train_X.tail(1).index, type(train_X.tail(1).index))
        # #     # logger.debug('s.index', s.index, type(s.index))
        # #     # logger.debug(train_X, type(train_X))
        # #     # logger.debug(s, type(s))
        # #     logger.debug('model', type(model))
        # #     logger.debug('model.predict', type(model.predict))
        # #     # exit()
        # #     preds = model.predict(s, objective=None, X_train=train_X, y_train=y_train).values
        # #     # except Exception as e:
        # #     #     logger.error(e)
        # #     #     logger.error('EvalML failed during prediction')
        # #     #     break
        # #     logger.debug('preds.shape', preds.shape)
        # #     assert len(preds) == horizon
        # #     predictions.append(preds)
        # #     train_X = pd.concat([train_X, s])

        # # # Flatten predictions and truncate if needed
        # # logger.debug('len(predictions)', len(predictions))
        # # logger.debug('predictions[0]', predictions[0].shape)
        # # predictions = np.concatenate([ p.flatten() for p in predictions ])
        # # logger.debug('predictions', predictions.shape)
        # # predictions = predictions[:len(test_X)]
        # # logger.debug('predictions', predictions.shape)
        # # return predictions
