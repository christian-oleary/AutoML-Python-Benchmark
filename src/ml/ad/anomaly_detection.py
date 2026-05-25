"""AutoML Anomaly Detection."""

from loguru import logger
import pandas as pd
from pyod.models.lunar import LUNAR
from pyod.models.ts_od import TimeSeriesOD

try:  # PyCaret
    from pycaret.anomaly import create_model, predict_model, setup
except ImportError as e:
    logger.warning(f'PyCaret is not installed: {e}')


class BaseADModel:
    """Base class for anomaly detection models."""

    def fit(self, df: pd.DataFrame, target_col: str = 'anomaly', **kwargs):
        """Drop the target column before calling _fit().

        :param pd.DataFrame df: The input DataFrame containing the features and target column.
        :param str target_col: The name of the target column to drop before fitting.
        """
        if isinstance(df, pd.DataFrame) and target_col in df.columns:
            df = df[[col for col in df.columns if col != target_col]]
        self._fit(df, **kwargs)

    def decision_function(self, df: pd.DataFrame):
        """Get anomaly scores for the data."""
        return self.predict_proba(df)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict anomalies in the data."""
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Get anomaly scores for the data."""
        raise NotImplementedError

    def _fit(self, X: pd.DataFrame, **kwargs) -> None:
        """Run PyCaret's unsupervised anomaly detection."""
        raise NotImplementedError


class PyCaretADModel(BaseADModel):
    """Anomaly Detection using PyCaret.

    :param str model_name: The name of the PyCaret anomaly detection model to use.
    :param float contamination: The proportion of anomalies in the dataset.
    """

    parameter_options = {
        'model_name': [
            'abod',  # Angle-base Outlier Detection
            'cluster',  # Clustering-Based Local Outlier
            'cof',  # Connectivity-Based Outlier Factor
            'histogram',  # Histogram-based Outlier Detection
            'iforest',  # Isolation Forest
            'knn',  # k-Nearest Neighbors Detector
            'lof',  # Local Outlier Factor
            'svm',  # One-class SVM detector
            # # 'pca',  # Principal Component Analysis  # Convergence errors
            # # 'mcd',  # Minimum Covariance Determinant  # > 12 hours
            # # 'sod',  # Subspace Outlier Detection  # Unstable
            'sos',  # Stochastic Outlier Selection
        ]
    }

    def __init__(self, model_name: str, contamination: float = 0.05, **kwargs):
        # Validate model name
        if model_name not in self.parameter_options['model_name']:
            raise ValueError(
                f'Model "{model_name}" is not allowed. '
                f'Options: {self.parameter_options["model_name"]}'
            )
        self.model_name = model_name
        self.contamination = contamination
        self.verbose = kwargs.get('verbosity', 1)
        self.model = None

    def fit(self, df: pd.DataFrame, target_col: str = 'anomaly', **kwargs):
        """Fit the PyCaret anomaly detection model.

        :param pd.DataFrame df: The input DataFrame containing the features and target column.
        :param str target_col: The name of the target column to drop before fitting.
        """
        self.contamination = kwargs.get('contamination', self.contamination)
        super().fit(df, target_col=target_col, **kwargs)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict anomalies using the fitted PyCaret model.

        :param pd.DataFrame X: The input DataFrame containing the features to predict on.
        :return pd.Series: A Series containing the predicted anomaly labels.
        """
        if self.model is None:
            raise ValueError(f'PyCaret {self.model_name} not been fitted yet.')
        result = predict_model(self.model, data=X)
        self.decision_scores_ = result['Anomaly_Score']
        return result['Anomaly']

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Get anomaly scores for the data.

        :param pd.DataFrame X: The input DataFrame containing the features to predict on.
        :return: A Series containing the anomaly scores.
        """
        if self.model is None:
            raise ValueError(f'PyCaret {self.model_name} not been fitted yet.')
        result = predict_model(self.model, data=X)
        self.decision_scores_ = result['Anomaly_Score']
        return result['Anomaly_Score']

    def _fit(self, X: pd.DataFrame, **kwargs) -> None:
        """Run PyCaret's unsupervised anomaly detection.

        :param pd.DataFrame X: The input DataFrame.
        :param str model_name: The name of the PyCaret anomaly detection model to use.
        :param float contamination: The proportion of anomalies in the dataset.
        """
        model_name = kwargs.get('model_name', self.model_name)
        self.contamination = kwargs.get('contamination', self.contamination)

        # PyCaret setup
        logger.debug('Running PyCaret setup...')
        kwargs = {
            'data': X,
            'normalize': True,
            'session_id': 1,
            'use_gpu': True,
            'verbose': (self.verbose - 1) > 1,  # PyCaret expects a boolean
        }
        try:
            setup(**kwargs)  # pylint: disable=unexpected-keyword-arg
        except TypeError:
            del kwargs['use_gpu']  # Older PyCaret versions don't support use_gpu
            setup(**kwargs)  # pylint: disable=unexpected-keyword-arg

        # Train model
        logger.debug(f'Fitting {model_name}...')
        self.model = create_model(model_name, fraction=self.contamination, verbose=self.verbose > 1)


class TimeSeriesODModel(TimeSeriesOD):
    """Time Series Anomaly Detection using PyOD's TimeSeriesOD model.

    :param str model_name: The PyOD anomaly detection model name or object to use, default is 'IForest'.
    :param float contamination: The proportion of anomalies in the dataset; default is 0.1.
    :param str score_aggregation: The method to aggregate anomaly scores over the sliding window; default is 'max'.
    :param int step: The step size for the sliding window; default is 1.
    :param int window_size: The size of the sliding window; default is 50.
    """

    parameter_options = {'model_name': ['IForest']}

    def __init__(
        self,
        model_name='IForest',
        window_size=50,
        step=1,
        score_aggregation='max',
        contamination=0.1,
        **_,
    ):
        self.model_name = model_name
        super().__init__(model_name, window_size, step, score_aggregation, contamination)

    def fit(self, X, y=None, **kwargs):
        """Fit the LUNAR model."""
        super().fit(X, y, **kwargs)
        if not hasattr(self, 'detector') and hasattr(self, 'detector_'):
            self.detector = self.detector_
        self.decision_scores_ = self.detector.decision_function(X)

    def decision_function(self, X):
        """Get anomaly scores for the data."""
        self.decision_scores_ = self.detector.decision_function(X)
        return self.decision_scores_


class LunarADModel(TimeSeriesOD):
    """LUNAR Anomaly Detection using a graph neural network-based method.

    LUNAR paper: https://www.aaai.org/AAAI22Papers/AAAI-51.GoodgeA.pdf
    From original LUNAR repo: https://github.com/agoodge/LUNAR
    "[LUNAR involves] unifying local outlier methods by showing that they are particular cases of
    the more general message passing framework used in graph neural networks. This allows us to
    introduce learnability into local outlier methods, in the form of a neural network [...] we
    propose LUNAR, a [...] graph neural network-based anomaly detection method. LUNAR learns to use
    information from the nearest neighbours of each node in a trainable way to find anomalies.

    LUNAR parameters in PyOD:
    model_type='WEIGHT', n_neighbours=5, negative_sampling='MIXED', val_size=0.1,
    scaler=MinMaxScaler(), epsilon=0.1, proportion=1.0, n_epochs=200, lr=0.001,
    wd=0.1, verbose=0, contamination=0.1, algorithm='auto', leaf_size=30,
    metric='minkowski', p=2, metric_params=None, n_jobs=1, **kwargs
    """

    parameter_options = {
        'model_name': [  # n_epochs
            'E-1_N-5',  # 1 epoch, 5 neighbours
            'E-2_N-5',  # 2 epochs, 5 neighbours
            'E-4_N-5',  # 4 epochs, 5 neighbours
            'E-8_N-5',  # 8 epochs, 5 neighbours
            'E-16_N-5',  # 16 epochs, 5 neighbours
            'E-4_N-2',  # 4 epochs, 2 neighbours
            'E-4_N-4',  # 4 epochs, 4 neighbours
            'E-4_N-8',  # 4 epochs, 8 neighbours
            'E-4_N-16',  # 4 epochs, 16 neighbours
        ]
    }

    def __init__(
        self, model_name='LUNAR', contamination=0.1, model_type='SCORE', window_size=50, **kwargs
    ):
        self.model_name = model_name
        if model_name == 'LUNAR':
            n_epochs = 10
            n_neighbours = 5
        else:
            n_epochs = int(self.model_name.split('_')[0].split('-')[1])
            n_neighbours = int(self.model_name.split('_')[1].split('-')[1])

        kwargs['verbose'] = kwargs.get('verbosity', 1) - 1
        lunar_model = LUNAR(
            model_type=model_type,
            n_neighbours=n_neighbours,
            n_epochs=n_epochs,
            contamination=contamination,
            **kwargs,
        )
        super().__init__(
            detector=lunar_model,  # type: ignore
            window_size=window_size,
            contamination=contamination,
        )

    def fit(self, X, y=None, **kwargs):
        """Fit the LUNAR model."""
        super().fit(X, y, **kwargs)
        if hasattr(self, 'detector') and hasattr(self, 'detector_'):
            self.detector = self.detector_
        self.decision_scores_ = self.detector.decision_scores_

    def decision_function(self, X):
        """Get anomaly scores for the data."""
        self.decision_scores_ = self.detector.decision_function(X)
        return self.decision_scores_
