"""Module for calculating and storing evaluation metrics for ML models."""

from __future__ import annotations

import csv
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import ConstantInputWarning, gmean, pearsonr, spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

from ml.logs import logger

# try:
#     from pandas.errors import IndexingError
# except ImportError:  # Older pandas
#     from pandas.core.indexing import IndexingError


class Result(BaseModel):
    """Class to hold the results of the evaluation metrics."""

    actual: np.ndarray
    predicted: np.ndarray

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Convert pd.Series to NumPy Array
        if self.predicted.shape == (self.actual.shape[0], 1) and not pd.core.frame.DataFrame:
            self.predicted = self.predicted.flatten()

        # Check data shapes
        if self.predicted.shape != self.actual.shape:
            raise ValueError(
                f'Predicted ({self.predicted.shape}) and actual '
                f'({self.actual.shape}) shapes do not match!'
            )


class RegressionResult(Result):
    """Class to hold regression results."""

    multioutput: str = 'uniform_average'
    y_train: np.ndarray | None = None

    duration: float | None = None
    mae: float | None
    mae_over: float | None
    mae_under: float | None
    mape: float | None
    mase: float | None
    me: float | None
    mse: float | None
    pearson_corr: float | None
    pearson_pvalue: float | None
    r2: float | None
    rmse: float | None
    smape: float | None
    spearman_corr: float | None
    spearman_pvalue: float | None

    def __init__(self, **data):
        super().__init__(**data)

        # Calculate metrics
        self.mae = mean_absolute_error(self.actual, self.predicted, multioutput=self.multioutput)
        self.mae_over = Metrics.mae_over(self.actual, self.predicted)
        self.mae_under = Metrics.mae_under(self.actual, self.predicted)
        self.mape = mean_absolute_percentage_error(
            self.actual, self.predicted, multioutput=self.multioutput
        )
        if self.y_train is not None:
            self.mase = MeanAbsoluteScaledError(multioutput=self.multioutput)(
                self.actual, self.predicted, y_train=self.y_train
            )
        self.me = np.mean(self.actual - self.predicted)
        self.mse = mean_squared_error(self.actual, self.predicted)
        self.rmse = math.sqrt(self.mse)
        self.r2 = r2_score(self.actual, self.predicted)
        self.smape = Metrics.smape(self.actual, self.predicted)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ConstantInputWarning)
            self.pearson_corr, self.pearson_pvalue = Metrics.correlation(
                self.actual, self.predicted, method='pearson'
            )
            self.spearman_corr, self.spearman_pvalue = Metrics.correlation(
                self.actual, self.predicted, method='spearman'
            )


class Metrics:
    """Utility functions."""

    # Ignored AutoML library presets
    ignored_presets: list[str] = []

    @staticmethod
    def regression_scores(
        actual: np.ndarray,
        predicted: np.ndarray,
        y_train: np.ndarray,
        scores_dir: str | Path | None = None,
        library_name: str | None = None,
        multioutput: str = 'uniform_average',
    ) -> dict:
        """Calculate forecasting metrics and optionally save results.

        :param np.ndarray actual: Original time series values
        :param np.ndarray predicted: Predicted time series values
        :param np.ndarray y_train: Training values (required for MASE)
        :param str scores_dir: Path to file to record scores (str or None), defaults to None
        :param str library_name: Name of model or library
        :param str multioutput: 'raw_values', 'uniform_average', defaults to 'uniform_average'
        :raises TypeError: If library_name is not provided when saving results to file
        :return dict: Dictionary of forecasting metrics
        """
        result = RegressionResult(
            actual=actual, predicted=predicted, y_train=y_train, multioutput=multioutput
        )
        scores = result.model_dump()
        logger.debug('Regression scores: ' + str(scores))
        # Save scores to file
        if scores_dir is not None:
            # Ensure library/model name is provided
            if library_name is None:
                raise TypeError('Library/model name required to save scores')
            # Create directory if it does not exist
            os.makedirs(scores_dir, exist_ok=True)
            # Save results to file
            Metrics.write_to_csv(os.path.join(scores_dir, f'{library_name}.csv'), scores)
        return scores

    @staticmethod
    def geometric_mean(error_score: float, rank_correlation_score: float) -> np.ndarray:
        """Calculates the geometric mean of some mean error score and a mean rank correlation score.

        :param float error_score: Mean error score
        :param float rank_score: Mean rank correlation score
        :return float: Geometric mean of error and 1-rank scores
        """
        # Grimes calls for "maximizing the geometric mean of (−MAE) and average daily Spearman correlation"
        # It is not possible to calculate geometric mean with negative numbers without some conversion.
        # Therefore, this work uses geometric mean of MAE and (1-SRC) with the intention of minimizing the metric.
        return gmean([error_score, 1 - rank_correlation_score])

    @staticmethod
    def geometric_mean_mae_sr(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        """Calculate geometric mean of MAE and a Spearman correlation score.

        :param np.ndarray actual: Real values
        :param np.ndarray predicted: Predicted values
        :return float: Geometric mean of MAE and SRC
        """
        mae_score = mean_absolute_error(actual, predicted, multioutput='uniform_average')
        src_score = Metrics.correlation(actual, predicted, method='spearman')[0]
        return Metrics.geometric_mean(mae_score, src_score)  # type: ignore

    @staticmethod
    def correlation(
        actual: np.ndarray, predicted: np.ndarray, method: str = 'pearson'
    ) -> tuple[float, float]:
        """Calculate correlation between actual and predicted values.

        :param np.array actual: Actual values
        :param np.array predicted: Predicted values
        :param str method: Correlation type, defaults to 'pearson'
        :raises ValueError: If unknown correlation method is passed
        :return: Correlation (float) and pvalue (float)
        """
        if method == 'pearson':
            result = pearsonr(actual, predicted)
        elif method == 'spearman':
            result = spearmanr(actual, predicted)
        else:
            raise ValueError(f'Unknown correlation method: {method}')

        try:
            correlation = result.correlation
            pvalue = result.pvalue
        except AttributeError:
            correlation = result[0]
            pvalue = result[1]

        return correlation, pvalue

    @staticmethod
    def mae_over(actual, predicted):
        """Overestimated predictions (from Grimes et al. 2014)."""
        errors = predicted - actual
        positive_errors = np.clip(errors, 0, errors.max())
        return np.mean(positive_errors)

    @staticmethod
    def mae_under(actual, predicted):
        """Underestimated predictions (from Grimes et al. 2014)."""
        errors = predicted - actual
        negative_errors = np.clip(errors, errors.min(), 0)
        return np.absolute(np.mean(negative_errors))

    @staticmethod
    def smape(actual, predicted):
        """Implementation of sMAPE."""
        totals = np.abs(actual) + np.abs(predicted)
        differences = np.abs(predicted - actual)
        return 100 / len(actual) * np.sum(2 * differences / totals)
        # return (
        #     100
        #     / len(actual)
        #     * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))
        # )

    @staticmethod
    def write_to_csv(path, results):
        """Record modelling results in a CSV file.

        :param str path: the result file path
        :param dict results: a dict containing results from running a model
        """
        np.set_printoptions(precision=4)
        if len(results) > 0:
            col_names = sorted(list(results.keys()), key=lambda v: str(v).upper())
            if 'model' in col_names:
                col_names.insert(0, col_names.pop(col_names.index('model')))

            for key, value in results.items():
                if value is None or value == '':
                    results[key] = 'None'

            try:
                Metrics._write_to_csv(path, results, col_names)
            except OSError:
                # try a second time: permission error can be due to Python not
                # having closed the file fast enough after the previous write
                time.sleep(1)  # in seconds
                Metrics._write_to_csv(path, results, col_names)

    @staticmethod
    def _write_to_csv(path, results, headers):
        """Open and write results to CSV file.

        :param str path: Path to file
        :param dict results: Values to write
        :param list headers: A list of strings to order values by
        """
        is_new_file = not os.path.exists(path)
        with open(path, 'a+', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=',')
            if is_new_file:
                writer.writerow(headers)
            writer.writerow([results[header] for header in headers])
