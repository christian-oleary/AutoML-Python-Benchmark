"""
Utility functions
"""

import os
import logging
import math
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr, t
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, median_absolute_error,
                             mean_squared_error, r2_score)
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError


logger_name = 'Benchmark'
logger = logging.getLogger(logger_name)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(level=logging.DEBUG)


class Utils:
    """Utility functions"""

    logger = logging.getLogger(logger_name)


    @staticmethod
    def regression_scores(actual, predicted, multioutput='raw_values'):
        """Calculate forecasting errors and correlations between predictions and actual values

        :param actual: Original time series values
        :param predicted: Predicted time series values
        :param multioutput:  'raw_values' (raw errors), 'uniform_average' (averaged errors), defaults to 'raw_values'
        :return results: Dictionary of results
        """

        mase = MeanAbsoluteScaledError(multioutput='raw_values')
        results = {
            'MAE': mean_absolute_error(actual, predicted, multioutput=multioutput),
            'MAE2': median_absolute_error(actual, predicted),
            'MAPE': mean_absolute_percentage_error(actual, predicted, multioutput=multioutput),
            'MASE': mase(actual, predicted, multioutput=multioutput),
            'ME': np.mean(actual - predicted),
            'MSE': mean_squared_error(actual, predicted, multioutput=multioutput),
            'Pearson Correlation': pearsonr(actual, predicted).correlation,
            'Pearson P-value': pearsonr(actual, predicted).pvalue,
            'R2': r2_score(actual, predicted, multioutput=multioutput),
            'RMSE': math.sqrt(mean_squared_error(actual, predicted, multioutput=multioutput)),
            'sMAPE': Utils.smape(actual, predicted),
            'Spearman Correlation': spearmanr(actual, predicted).correlation,
            'Spearman P-value': spearmanr(actual, predicted).pvalue,
        }
        return results


    @staticmethod
    def smape(actual, predicted):
        return 100/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


    @staticmethod
    def get_csv_datasets(datasets_directory):
        """Fetch list of file names of CSV datasets

        :param datasets_directory: Path to datasets directory (str)
        :raises NotADirectoryError: If datasets_directory does not exist
        :raises IOError: If datasets_directory does not have CSV files
        :return: list of dataset file names
        """

        if not os.path.exists(datasets_directory):
            raise NotADirectoryError('Datasets direcotry path does not exist')

        csv_files = [
            f for f in os.listdir(datasets_directory)
            if f.endswith('csv') and '0_metadata' not in f
            ]

        if len(csv_files) == 0:
            raise IOError('No CSV files found')

        return csv_files
