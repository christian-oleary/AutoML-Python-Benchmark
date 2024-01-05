"""
Utility functions
"""

import csv
import math
import os
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gmean, pearsonr, spearmanr
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, median_absolute_error,
                             mean_squared_error, r2_score)
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

from src.logs import logger


class Utils:
    """Utility functions"""

    ignored_presets = [
        # 'preset-superfast__60_proc-1_limit-60.0', 'preset-superfast__1200_proc-1_limit-3.0',
        # 'preset-superfast__900_proc-1_limit-4.0', 'preset-superfast__600_proc-1_limit-6.0',
        # 'preset-superfast__600_proc-10_limit-6.0', 'preset-superfast__900_proc-10_limit-4.0',
        # 'preset-superfast__300_proc-1_limit-12.0', 'preset-superfast__300_proc-10_limit-12.0',
        # 'preset-fast__900_proc-1_limit-4.0', 'preset-fast__1200_proc-1_limit-3.0', 'preset-fast__600_proc-1_limit-6.0',
        # 'preset-fast__900_proc-10_limit-4.0', 'preset-fast__300_proc-1_limit-12.0',
        # 'preset-fast__600_proc-10_limit-6.0', 'preset-fast__60_proc-1_limit-60.0',
        # 'preset-fast__300_proc-10_limit-12.0', 'preset-fast_parallel__1200_proc-1_limit-3.0',
        # 'preset-default__900_proc-10_limit-4.0', 'preset-default__900_proc-1_limit-4.0',
        # 'preset-fast_parallel__600_proc-10_limit-6.0', 'preset-fast_parallel__900_proc-10_limit-4.0',
        # 'preset-default__1200_proc-1_limit-3.0', 'preset-default__600_proc-10_limit-6.0',
        # 'preset-fast_parallel__900_proc-1_limit-4.0', 'preset-fast_parallel__300_proc-10_limit-12.0',
        # 'preset-fast_parallel__600_proc-1_limit-6.0', 'preset-default__600_proc-1_limit-6.0',
        # 'preset-default__300_proc-10_limit-12.0', 'preset-fast_parallel__300_proc-1_limit-12.0',
        # 'preset-default__300_proc-1_limit-12.0', 'preset-fast_parallel__60_proc-1_limit-60.0',
        # 'preset-default__60_proc-1_limit-60.0', 'preset-all__600_proc-1_limit-6.0',
        # 'preset-all__900_proc-1_limit-4.0','preset-all__1200_proc-1_limit-3.0',
        # 'preset-all__300_proc-1_limit-12.0','preset-all__300_proc-10_limit-12.0',
        # 'preset-all_300_proc-1_limit-12.0', 'preset-all_600_proc-1_limit-6.0', 'preset-all_60_proc-1_limit-60.0',
        # 'preset-all__600_proc-10_limit-6.0', 'preset-all__900_proc-10_limit-4.0',
        # 'preset-default_300_proc-1_limit-12.0', 'preset-default_600_proc-1_limit-6.0',
        # 'preset-default_60_proc-1_limit-60.0', 'preset-all__60_proc-1_limit-60.0',
        ]


    @staticmethod
    def regression_scores(actual, predicted, y_train,
                          scores_dir=None,
                          forecaster_name=None,
                          multioutput='uniform_average',
                          **kwargs):
        """Calculate forecasting metrics and optionally save results.

        :param np.array actual: Original time series values
        :param np.array predicted: Predicted time series values
        :param np.array y_train: Training values (required for MASE)
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
            'MAEover': Utils.mae_over(actual, predicted),
            'MAEunder': Utils.mae_under(actual, predicted),
            'MAPE': mean_absolute_percentage_error(actual, predicted, multioutput=multioutput),
            'MASE': mase(actual, predicted, y_train=y_train),
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

        # Grimes calls for "maximizing the geometric mean of (−MAE) and average daily Spearman correlation"
        # This must be an error, as you cannot calculate geometric mean with negative numbers. This uses
        # geometric mean of MAE and (1-SRC) with the intention of minimizing the metric.
        results['GM-MAE-SR'] = gmean([results['MAE'], 1 - results['Spearman Correlation']])
        results['GM-MASE-SR'] = gmean([results['MASE'], 1 - results['Spearman Correlation']])

        if 'duration' in kwargs.keys():
            results['duration'] = kwargs['duration']

        if scores_dir != None:
            if forecaster_name == None:
                raise TypeError('Forecaster name required to save scores')
            os.makedirs(scores_dir, exist_ok=True)

            results = {
                **results,
                'environment': f'python_{platform.python_version()}-os_{platform.system()}',
                'device': f'node_{platform.node()}-pro_{platform.processor()}',
            }

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
    def mae_over(actual, predicted):
        """Overestimated predictions (from Grimes et al. 2014)"""
        errors = predicted - actual
        positive_errors = np.clip(errors, 0, errors.max())
        return np.mean(positive_errors)


    @staticmethod
    def mae_under(actual, predicted):
        """Underestimated predictions (from Grimes et al. 2014)"""
        errors = predicted - actual
        negative_errors = np.clip(errors, errors.min(), 0)
        return np.absolute(np.mean(negative_errors))


    @staticmethod
    def smape(actual, predicted):
        """sMAPE"""
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
    def save_plot(title,
                  xlabel=None,
                  ylabel=None,
                  suptitle='',
                  show=False,
                  legend=None,
                  save_path=None,
                  yscale='linear'):
        """Apply title and axis labels to plot. Show and save to file. Clear plot.

        :param title: Title for plot
        :param xlabel: Plot X-axis label
        :param ylabel: Plot Y-axis label
        :param title: Subtitle for plot
        :param show: Show plot on screen, defaults to False
        :param legend: Legend, defaults to None
        :param save_path: Save plot to file if not None, defaults to None
        :param yscale: Y-Scale ('linear' or 'log'), defaults to 'linear'
        """

        if xlabel != None:
            plt.xlabel(xlabel)

        if ylabel != None:
            plt.ylabel(ylabel)

        plt.yscale(yscale)

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
        plt.cla()
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

        # Save all scores as CSV
        output_file = os.path.join(stats_dir, '1_all_scores.csv')
        test_scores.to_csv(output_file, index=False)

        # Scores per library across all presets and failed training counts
        if len(test_scores) > 0:
            summarized_scores = Utils.save_latex(test_scores, output_file.replace('csv', 'tex'))
            if plots:
                Utils.save_heatmap(summarized_scores, os.path.join(stats_dir, 'heatmap.csv'),
                                   os.path.join(stats_dir, 'heatmap.png'))
                # Utils.plot_test_scores(test_scores, stats_dir, plots)


    @staticmethod
    def save_latex(df, output_file):
        """Save dataframe of results in a LaTeX file

        :param pd.DataFrame df: Results
        :param str output_file: Path to .tex file
        """
        # Sort by GM-MAE-SR
        df = df.sort_values('GM-MAE-SR')
        # Filter columns
        df = df[['library', 'preset', 'duration', 'GM-MAE-SR', 'MAE', 'MASE', 'MSE', 'RMSE',
                                        'Spearman Correlation']]
        # Rename columns
        df.columns = ['Library', 'Preset', 'Duration (sec.)', 'GM-MAE-SR', 'MAE', 'MASE', 'MSE', 'RMSE',
                                    'SRC']
        df['Library'] = df['Library'].str.capitalize() # Format library names
        df['Preset'] = df['Preset'] \
            .str.replace('Fedot', 'FEDOT') \
            .str.replace('Flaml', 'FLAML') \
            .str.replace('Autokeras', 'AutoKeras') \
            .str.replace('Autogluon', 'AutoGluon') \
            .str.replace('preset-', '') \
            .str.replace('proc-1', '') \
            .str.replace('proc-10', '') \
            .str.replace('-limit-3600', '') \
            .str.replace('-limit-3240', '') \
            .str.replace('-limit-3564', '') \
            .str.replace('-limit-57', '') \
            .str.replace('-limit-12', '') \
            .str.replace('-limit-60', '') \
            .str.replace('-limit-6', '') \
            .str.replace('_', ' ') \
            .str.capitalize() \
            .str.replace(' ', '-') \
            .str.replace('--', '-')

        # Save all scores as TEX
        df.style.format(precision=2, thousands=',', decimal='.').to_latex(
            output_file.replace('csv', 'tex'),
            caption='Test Scores Ordered by GM-MAE-SR',
            environment='table*',
            hrules=True,
            label='tab:summarized_scores',
            multirow_align='t',
            position='!htbp',
            )
        return df


    @staticmethod
    def plot_test_scores(test_scores, stats_dir, plots):
        """Plot test scores

        :param pd.DataFrame test_scores: Test scores
        :param str stats_dir: Path to output directory
        :param bool plots: If True, generate plots
        """
        test_scores['library'] = test_scores['library'].str.capitalize()

        # Ignore deprecated/unused presets
        test_scores = test_scores[~test_scores['preset'].isin(Utils.ignored_presets)]

        # Save overall scores and generate plots
        if plots:
            # Bar plot of failed training attempts
            test_scores.plot.bar(y='failed', figsize=(35, 10))
            save_path = os.path.join(stats_dir, '3_failed_counts.png')
            Utils.save_plot('Failed Counts', save_path=save_path)

            # Boxplots
            for col, filename, title in [
                ('MAE', '5_MAE_box.png', 'MAE'),
                ('MSE', '5_MSE_box.png', 'MSE'),
                ('RMSE', '6_RMSE_box.png', 'RMSE'),
                ('Spearman Correlation', '6_Spearman_Correlation_box.png', 'Spearman Correlation'),
                ('duration', '8_duration_box.png', 'Duration (sec)'),
                ]:
                test_scores.boxplot(col, by='library')
                save_path = os.path.join(stats_dir, filename)
                Utils.save_plot(title, save_path=save_path)

        # Save mean scores and generate plots
        df_failed = test_scores[['library', 'failed']]
        df_failed = df_failed.set_index('library')
        df_failed = df_failed.groupby('library').sum()

        def plot_averages(group_col, cols_to_drop, fig_width, fig_height):
            df_grouped = test_scores.drop(cols_to_drop, axis=1).groupby(group_col)

            df_grouped.index = df_grouped[group_col]
            df_max = df_grouped.max()
            df_min = df_grouped.min()
            df_mean = df_grouped.mean()

            df_max.columns = [ f'{c}_max' for c in df_max.columns.tolist() ]
            df_min.columns = [ f'{c}_min' for c in df_min.columns.tolist() ]
            df_mean.columns = [ f'{c}_mean' for c in df_mean.columns.tolist() ]

            mean_scores = pd.concat([df_failed, df_max, df_min, df_mean], axis=1)

            output_file = os.path.join(stats_dir, f'3_mean_scores_by_{group_col}.csv')
            mean_scores.to_csv(output_file)

            if plots:
                # Bar plot of failed training attempts
                mean_scores.plot.bar(y='failed', figsize=(fig_width, fig_height))
                save_path = os.path.join(stats_dir, f'3_failed_counts_by_{group_col}.png')
                Utils.save_plot(f'Failed Counts by {group_col}', save_path=save_path, legend=None, yscale='linear')

                # Boxplots
                for col, filename, title, yscale in [
                    ('R2_mean', f'4_R2_mean_by_{group_col}.png', 'Mean R2', 'linear'),
                    ('MAE_mean', f'5_MAE_mean_by_{group_col}.png', 'Mean MAE', 'linear'),
                    ('MSE_mean', f'6_MSE_mean_by_{group_col}.png', 'Mean MSE', 'linear'),
                    ('duration_mean', f'7_duration_mean_by_{group_col}.png', 'Mean Duration', 'linear'),
                    ]:
                    mean_scores.plot.bar(y=col, figsize=(fig_width, fig_height))
                    save_path = os.path.join(stats_dir, filename)
                    Utils.save_plot(title, save_path=save_path, legend=None, yscale=yscale)

        # Plot mean scores by library
        plot_averages(group_col='library',
                      cols_to_drop=['file', 'failed', 'preset'],
                      fig_width=6,
                      fig_height=3)

        # # Plot mean scores by library/preset
        # test_scores['library-preset'] = test_scores['library'] + ': ' + test_scores['preset']
        # plot_averages(group_col='library-preset',
        #               cols_to_drop=['file', 'failed', 'preset', 'library'],
        #               fig_width=35,
        #               fig_height=10)


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

            # Save overall scores as CSV and TEX
            overall_scores_path = os.path.join(stats_dir, '1_all_scores.csv')
            logger.debug(f'Compiling overall test scores in {overall_scores_path}')
            all_scores.to_csv(overall_scores_path, index=False)
            summarized_scores = Utils.save_latex(all_scores, overall_scores_path.replace('csv', 'tex'))


            if plots:
                logger.debug('Generating plots')
                Utils.save_heatmap(summarized_scores, os.path.join(stats_dir, 'heatmap.csv'),
                                   os.path.join(stats_dir, 'heatmap.png'))
                Utils.plot_test_scores(all_scores, stats_dir, plots)


    def save_heatmap(df, csv_path, png_path):
        """Save Pearson Correlation Matrix of metrics

        :param pd.DataFrame df: Results
        :param str csv_path: Path to CSV file
        :param str png_path: Path to PNG file
        """
        # Save Perason correlation heatmap of metrics as an indication of agreement.
        columns = ['GM-MAE-SR', 'MAE', 'MASE', 'MSE', 'RMSE', 'SRC']
        heatmap = df[columns].corr(method='pearson')
        heatmap.to_csv(csv_path)
        axes = sns.heatmap(heatmap,
                            annot=True,
                            cbar=False,
                            cmap='viridis',
                            fmt='.2f',
                        #    xticklabels=columns,
                        #    yticklabels=columns,
                            annot_kws={ 'size': 11 }
                            )
        axes.set_xticklabels(axes.get_xticklabels(), fontsize=11, rotation=45, ha='right')
        axes.set_yticklabels(axes.get_yticklabels(), fontsize=11, rotation=45, va='top')
        # axes.set_xticklabels(columns, fontsize=11, rotation=45, ha='right')
        # axes.set_yticklabels(columns, fontsize=11, rotation=45, va='top')
        plt.tight_layout()
        Utils.save_plot('Pearson Correlation Heatmap', save_path=png_path)
