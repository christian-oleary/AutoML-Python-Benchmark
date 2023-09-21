"""
Utility functions
"""

import csv
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, median_absolute_error,
                             mean_squared_error, r2_score)
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

from src.logs import logger


class Utils:
    """Utility functions"""


    @staticmethod
    def regression_scores(actual, predicted,
                          scores_dir=None,
                          forecaster_name=None,
                          multioutput='uniform_average',
                          **kwargs):
        """Calculate forecasting metrics and optionally save results.

        :param np.array actual: Original time series values
        :param np.array predicted: Predicted time series values
        :param str scores_dir: Path to file to record scores (str or None), defaults to None
        :param str forecaster_name: Name of model (str)
        :param str multioutput: 'raw_values' (raw errors), 'uniform_average' (averaged errors), defaults to 'uniform_average'
        :raises TypeError: If forecaster_name is not provided when saving results to file
        :return results: Dictionary of results
        """

        # Convert pd.Series to NumPy Array
        if predicted.shape == (actual.shape[0], 1) and not pd.core.frame.DataFrame:
            predicted = predicted.flatten()

        if predicted.shape != actual.shape:
            raise ValueError(f'Predicted ({predicted.shape}) and actual ({actual.shape}) shapes do not match!')

        mase = MeanAbsoluteScaledError(multioutput='uniform_average')
        pearson = Utils.correlation(actual, predicted, method='pearson')
        spearman = Utils.correlation(actual, predicted, method='spearman')

        results = {
            'MAE': mean_absolute_error(actual, predicted, multioutput=multioutput),
            'MAE2': median_absolute_error(actual, predicted),
            'MAPE': mean_absolute_percentage_error(actual, predicted, multioutput=multioutput),
            # 'MASE': mase(actual, predicted),
            'ME': np.mean(actual - predicted),
            'MSE': mean_squared_error(actual, predicted, multioutput=multioutput),
            'Pearson Correlation': pearson[0],
            'Pearson P-value': pearson[1],
            'R2': r2_score(actual, predicted, multioutput=multioutput),
            'RMSE': math.sqrt(mean_squared_error(actual, predicted, multioutput=multioutput)),
            'sMAPE': Utils.smape(actual, predicted),
            'Spearman Correlation': spearman[0],
            'Spearman P-value': spearman[1],
        }

        if 'duration' in kwargs.keys():
            results['duration'] = kwargs['duration']

        if scores_dir != None:
            if forecaster_name == None:
                raise TypeError('Forecaster name required to save scores')
            os.makedirs(scores_dir, exist_ok=True)
            Utils.write_to_csv(os.path.join(scores_dir, f'{forecaster_name}.csv'), results)

        return results


    def correlation(actual, predicted, method='pearson'):
        """Wrapper to extract correlations and p-values from scipy

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
        except: # older scipy versions returned a tuple instead of an object
            correlation = result[0]
            pvalue = result[1]

        return correlation, pvalue


    @staticmethod
    def smape(actual, predicted):
        return 100/len(actual) * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


    @staticmethod
    def write_to_csv(path, results):
        """Record modelling results in a CSV file.

        :param str path: the result file path
        :param dict results: a dict containing results from running a model
        """

        np.set_printoptions(precision=4)

        # Remove unneeded values
        # unused_cols = []
        # for col in unused_cols:
        #     if col in results.keys():
        #         del results[col]

        if len(results) > 0:
            HEADERS = sorted(list(results.keys()), key=lambda v: str(v).upper())
            if 'model' in HEADERS:
                HEADERS.insert(0, HEADERS.pop(HEADERS.index('model')))

            for key, value in results.items():
                if value == None or value == '':
                    results[key] = 'None'

            try:
                Utils._write_to_csv(path, results, HEADERS)
            except OSError as _:
                # try a second time: permission error can be due to Python not
                # having closed the file fast enough after the previous write
                time.sleep(1) # in seconds
                Utils._write_to_csv(path, results, HEADERS)


    @staticmethod
    def _write_to_csv(path, results, headers):
        """Open and write results to CSV file.

        :param str path: Path to file
        :param dict results: Values to write
        :param list headers: A list of strings to order values by
        """

        is_new_file = not os.path.exists(path)
        with open(path, 'a+', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            if is_new_file:
                writer.writerow(headers)
            writer.writerow([results[header] for header in headers])


    @staticmethod
    def plot_forecast(actual, predicted, results_subdir, forecaster_name):
        """Plot forecasted vs actual values

        :param np.array actual: Original time series values
        :param np.array predicted: Forecasted values
        :param str results_subdir: Path to output directory
        :param str forecaster_name: Model name
        """
        plt.clf()
        pd.plotting.register_matplotlib_converters()

        # Create plot
        plt.figure(0, figsize=(20, 3)) # Pass plot ID to prevent memory issues
        plt.plot(actual, label='actual')
        plt.plot(predicted, label='predicted')
        save_path = os.path.join(results_subdir, 'plots', f'{forecaster_name}.png')
        os.makedirs(os.path.join(results_subdir, 'plots'), exist_ok=True)
        Utils.save_plot(forecaster_name, save_path=save_path)


    @staticmethod
    def save_plot(title, xlabel=None, ylabel=None, suptitle='', show=False, legend=None, save_path=None):
        """Apply title and axis labels to plot. Show and save to file. Clear plot.

        :param title: Title for plot
        :param xlabel: Plot X-axis label
        :param ylabel: Plot Y-axis label
        :param show: Show plot on screen, defaults to False
        :param save_path: Save plot to file if not None, defaults to None
        """

        if xlabel != None:
            plt.xlabel(xlabel)

        if ylabel != None:
            plt.ylabel(ylabel)

        plt.title(title)
        plt.suptitle(suptitle)
        if legend != None:
            plt.legend(legend, loc='upper left')

        # Show plot
        if show:
            plt.show()
        # Show plot as file
        if save_path != None:
            plt.savefig(save_path, bbox_inches='tight')
        # Clear for next plot
        plt.clf()
        plt.close('all')


    @staticmethod
    def get_csv_datasets(datasets_directory):
        """Fetch list of file names of CSV datasets

        :param str datasets_directory: Path to datasets directory
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


    @staticmethod
    def split_test_set(test_df, horizon):
        """Split test dataset into list of smaller sets for rolling origin forecasting

        :param pd.DataFrame test_df: Test dataset
        :param int horizon: Forecasting horizon
        :return: List of DataFrame objects
        """

        test_splits = []
        total = 0 # total length of test splits
        for _ in range(0, len(test_df)-1, horizon): # The -1 is because the last split may be less than horizon
            try:
                test_splits.append(test_df.iloc[total:total+horizon, :])
            except: # If 1D (series)
                test_splits.append(test_df.iloc[total:total+horizon])
            total += horizon

        # Leftover rows
        if total < len(test_df):
            test_splits.append(test_df.tail(len(test_df) - total))
        return test_splits


    @staticmethod
    def summarize_dataset_results(results_dir, plots=True):
        """Analyse results saved to file

        :param str results_subdir: Path to relevant results directory
        :param bool plots: Save plots as images, defaults to True
        """

        stats_dir = os.path.join(results_dir, 'statistics')

        test_results = []
        failed = []
        dataset = os.path.basename(os.path.normpath(results_dir))

        # For each library/preset, get mean scores
        for library in os.listdir(results_dir):
            subdir = os.path.join(results_dir, library)
            for preset in os.listdir(subdir):
                preset_dir = os.path.join(subdir, preset)
                if subdir == stats_dir:
                    continue

                scores_path = os.path.join(preset_dir, f'{library}.csv')
                failed_path = os.path.join(preset_dir, f'failed.txt')

                if os.path.exists(scores_path):
                    df = pd.read_csv(scores_path, index_col=False)
                    test_results.append({'library': library, 'preset': preset, 'file': dataset, 'failed': 0,
                                         **df.mean(numeric_only=True).to_dict() })
                elif os.path.exists(failed_path):
                    failed.append({'library': library, 'preset': preset, 'file': dataset, 'failed': 1})
                else:
                    raise FileNotFoundError(f'Results file(s) missing in {preset_dir}')

        os.makedirs(stats_dir, exist_ok=True)

        # Combine scores into one CSV file
        test_scores = pd.DataFrame(test_results)
        if len(test_scores) > 0:
            failed = pd.DataFrame(failed)
            test_scores = pd.concat([test_scores, failed])

        output_file = os.path.join(stats_dir, '1_all_scores.csv')
        test_scores.to_csv(output_file, index=False)

        # Scores per library across all presets and failed training counts
        if len(test_scores) > 0:
            Utils.plot_test_scores(test_scores, stats_dir, plots)


    @staticmethod
    def plot_test_scores(test_scores, stats_dir, plots):
        """Plot test scores

        :param pd.DataFrame test_scores: Test scores
        :param str stats_dir: Path to output directory
        :param bool plots: If True, generate plots
        """
        test_scores['library'] = test_scores['library'].str.capitalize()

        # Save overall scores and generate plots
        if plots:
            # Bar plot of failed training attempts
            test_scores.plot.bar(y='failed')
            save_path = os.path.join(stats_dir, '3_failed_counts.png')
            Utils.save_plot('Failed Counts', save_path=save_path)

            # Boxplots
            for col, filename, title in [
                ('R2', '4_R2_box.png', 'R2'),
                ('MAE', '5_MAE_box.png', 'MAE'),
                ('MAPE', '6_MAPE_box.png', 'MAPE'),
                ('duration', '7_duration_box.png', 'Duration (sec)'),
                ]:
                test_scores.boxplot(col, by='library')
                save_path = os.path.join(stats_dir, filename)
                Utils.save_plot(title, save_path=save_path)

        # Save mean scores and generate plots
        df_failed = test_scores[['library', 'failed']]
        df_failed = df_failed.set_index('library')
        df_failed = df_failed.groupby('library').sum()

        df_by_library = test_scores.drop(['preset', 'file', 'failed'], axis=1).groupby('library')
        df_by_library.index = df_by_library['library']
        df_max = df_by_library.max()
        df_min = df_by_library.min()
        df_mean = df_by_library.mean()

        df_max.columns = [ f'{c}_max' for c in df_max.columns.tolist() ]
        df_min.columns = [ f'{c}_min' for c in df_min.columns.tolist() ]
        df_mean.columns = [ f'{c}_mean' for c in df_mean.columns.tolist() ]

        mean_scores = pd.concat([df_failed, df_max, df_min, df_mean], axis=1)

        output_file = os.path.join(stats_dir, '2_mean_scores.csv')
        mean_scores.to_csv(output_file)

        if plots:
            # Bar plot of failed training attempts
            # TODO: Need to divide by number of presets
            mean_scores.plot.bar(y='failed')
            save_path = os.path.join(stats_dir, '3_failed_counts.png')
            Utils.save_plot('Failed Counts', save_path=save_path)

            # Boxplots
            for col, filename, title in [
                ('R2_mean', '4_R2_mean.png', 'Mean R2'),
                ('MAE_mean', '5_MAE_mean.png', 'Mean MAE'),
                ('MAPE_mean', '6_MAPE_mean.png', 'Mean MAPE'),
                ('duration_mean', '7_duration_mean.png', 'Mean Duration'),
                ]:
                mean_scores.plot.bar(y=col)
                save_path = os.path.join(stats_dir, filename)
                Utils.save_plot(title, save_path=save_path)


    @staticmethod
    def summarize_overall_results(results_dir, forecast_type, plots=True):
        """Analyse results saved to file

        :param str results_subdir: Path to relevant results directory
        :param bool plots: Save plots as images, defaults to True
        """

        dataframes = []
        results_subdir = os.path.join(results_dir, f'{forecast_type}_forecasting')
        for dirpath, _, filenames in os.walk(results_subdir):
            if 'statistics' in dirpath and len(filenames) > 0:
                all_scores_path = os.path.join(dirpath, '1_all_scores.csv')
                try:
                    df = pd.read_csv(all_scores_path)
                    dataframes.append(df)
                except pd.errors.EmptyDataError as _:
                    logger.debug(f'No data found in {all_scores_path}. Skipping')

        all_scores = pd.concat(dataframes, axis=0)

        if len(all_scores) == 0:
            logger.warning('No results found. Skipping')
        else:
            stats_dir = os.path.join(results_dir, f'{forecast_type}_statistics')
            os.makedirs(stats_dir, exist_ok=True)

            logger.debug(f'Compiling test scores in {all_scores_path}')
            all_scores_path = os.path.join(stats_dir, '1_all_scores.csv')
            all_scores.to_csv(all_scores_path, index=False)

            if plots:
                logger.debug('Generating plots')
                Utils.plot_test_scores(all_scores, stats_dir, plots)
