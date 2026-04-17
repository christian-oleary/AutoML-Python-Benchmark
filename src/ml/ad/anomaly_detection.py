"""AutoML Anomaly Detection."""

from loguru import logger
import pandas as pd

try:  # PyCaret
    from pycaret.anomaly import create_model, predict_model, setup
except ImportError:
    logger.warning('PyCaret is not installed.')


class BaseADModel:
    """Base class for anomaly detection models."""

    def fit(self, df: pd.DataFrame, target_col: str = 'anomaly', **kwargs):
        """Drop the target column before calling _fit().

        :param pd.DataFrame df: The input DataFrame containing the features and target column.
        :param str target_col: The name of the target column to drop before fitting.
        """
        df = df[[col for col in df.columns if col != target_col]]
        self._fit(df, **kwargs)

    def predict(self, df: pd.DataFrame):
        """Predict anomalies in the data."""
        raise NotImplementedError

    def predict_proba(self, df: pd.DataFrame):
        """Get anomaly scores for the data."""
        raise NotImplementedError

    def decision_function(self, df: pd.DataFrame):
        """Get anomaly scores for the data."""
        return self.predict_proba(df)

    def _fit(self, df: pd.DataFrame, **kwargs) -> None:
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
            'pca',  # Principal Component Analysis
            # # 'mcd',  # Minimum Covariance Determinant  # > 12 hours
            # 'sod',  # Subspace Outlier Detection
            'sos',  # Stochastic Outlier Selection
        ]
    }

    def __init__(self, model_name: str, contamination: float = 0.05, **_):
        # Validate model name
        if model_name not in self.parameter_options['model_name']:
            raise ValueError(
                f'Model "{model_name}" is not allowed. '
                f'Options: {self.parameter_options["model_name"]}'
            )
        self.model_name = model_name
        self.contamination = contamination
        self.model = None

    def fit(self, df: pd.DataFrame, target_col: str = 'anomaly', **kwargs):
        """Fit the PyCaret anomaly detection model.

        :param pd.DataFrame df: The input DataFrame containing the features and target column.
        :param str target_col: The name of the target column to drop before fitting.
        """
        logger.debug(f'Fitting PyCaret {self.model_name}...')
        self.contamination = kwargs.get('contamination', self.contamination)
        super().fit(df, target_col=target_col, **kwargs)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict anomalies using the fitted PyCaret model.

        :param pd.DataFrame df: The input DataFrame containing the features to predict on.
        :return pd.Series: A Series containing the predicted anomaly labels.
        """
        if self.model is None:
            raise ValueError(f"PyCaret {self.model_name} not been fitted yet.")
        logger.debug(f'Predicting with PyCaret {self.model_name}...')
        result = predict_model(self.model, data=df)
        return result['Anomaly']

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """Get anomaly scores for the data.

        :param pd.DataFrame df: The input DataFrame containing the features to predict on.
        :return: A Series containing the anomaly scores.
        """
        if self.model is None:
            raise ValueError(f"PyCaret {self.model_name} not been fitted yet.")
        logger.debug(f'Getting anomaly scores with PyCaret {self.model_name}...')
        result = predict_model(self.model, data=df)
        return result['Anomaly_Score']

    def _fit(self, df, **kwargs) -> None:
        """Run PyCaret's unsupervised anomaly detection.

        :param pd.DataFrame df: The input DataFrame.
        :param str model_name: The name of the PyCaret anomaly detection model to use.
        :param float contamination: The proportion of anomalies in the dataset.
        """
        model_name = kwargs.get('model_name', self.model_name)
        self.contamination = kwargs.get('contamination', self.contamination)

        # PyCaret setup
        logger.debug('Running PyCaret setup...')
        kwargs = {'data': df, 'normalize': True, 'session_id': 1, 'use_gpu': True, 'verbose': True}
        try:
            setup(**kwargs)  # pylint: disable=unexpected-keyword-arg
        except TypeError:
            del kwargs['use_gpu']  # Older PyCaret versions don't support use_gpu
            setup(**kwargs)  # pylint: disable=unexpected-keyword-arg

        # Train model
        logger.debug(f'Fitting {model_name}...')
        self.model = create_model(model_name, fraction=self.contamination)
