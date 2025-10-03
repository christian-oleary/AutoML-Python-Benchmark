"""Plotting functions."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from pandas.errors import IndexingError
except ImportError:  # Older pandas
    from pandas.core.indexing import IndexingError

from ml.logs import logger
from ml.utils import Utils


class Plotter:
    """Plotting functions."""

    @staticmethod
    def plot_forecast(actual, predicted, results_subdir, library_name):
        """Plot forecasted vs actual values

        :param np.array actual: Original time series values
        :param np.array predicted: Forecasted values
        :param str results_subdir: Path to output directory
        :param str library_name: Library/model name
        """
        plt.clf()
        pd.plotting.register_matplotlib_converters()
        # Create plot
        plt.figure(0, figsize=(20, 3))  # Pass plot ID to prevent memory issues
        plt.plot(actual, label='actual')
        plt.plot(predicted, label='predicted')
        save_path = Path(results_subdir, 'plots', f'{library_name}.svg')
        Path(results_subdir, 'plots').mkdir(parents=True, exist_ok=True)
        Utils.save_plot(library_name, save_path=save_path)

    @staticmethod
    def save_plot(
        title,
        xlabel=None,
        ylabel=None,
        suptitle='',
        show=False,
        legend=None,
        save_path=None,
        yscale='linear',
    ):
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
        if xlabel is not None:
            plt.xlabel(xlabel)

        if ylabel is not None:
            plt.ylabel(ylabel)
        plt.yscale(yscale)

        plt.title(title)
        plt.suptitle(suptitle)
        if legend is not None:
            plt.legend(legend, loc='upper left')

        # Show plot
        if show:
            plt.show()
        # Show plot as file
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
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
            raise NotADirectoryError('Datasets directory path does not exist')

        csv_files = [
            f for f in os.listdir(datasets_directory) if f.endswith('csv') and '0_metadata' not in f
        ]
        if len(csv_files) == 0:
            raise IOError('No CSV files found')

        return csv_files

    @staticmethod
    def split_test_set(test_df: pd.DataFrame, horizon: int) -> list[pd.DataFrame]:
        """Split test dataset into list of smaller sets for rolling origin forecasting

        :param pd.DataFrame test_df: Test dataset
        :param int horizon: Forecasting horizon
        :return: List of DataFrame objects
        """
        test_splits = []
        total = 0  # total length of test splits
        for _ in range(0, len(test_df) - 1, horizon):  # -1 because last split may be < horizon
            try:
                test_splits.append(test_df.iloc[total : total + horizon, :])
            except IndexingError:  # If 1D (series)
                test_splits.append(test_df.iloc[total : total + horizon])
            total += horizon

        # Leftover rows
        if total < len(test_df):
            test_splits.append(test_df.tail(len(test_df) - total))
        return test_splits

    @staticmethod
    def summarize_dataset_results(results_dir: str | Path, plots: bool = True):
        """Analyse results saved to file

        :param str | Path results_subdir: Path to relevant results directory
        :param bool plots: Save plots as images, defaults to True
        """
        stats_dir = os.path.join(results_dir, 'statistics')

        test_results: list[dict] = []  # List of test results
        failed: list[dict] = []  # List of failed training attempts
        dataset = os.path.basename(os.path.normpath(results_dir))

        # For each library/preset, get mean scores
        test_results, failed = Utils.get_averaged_results(
            results_dir,
            stats_dir,
            test_results,
            dataset,
            failed,
        )

        os.makedirs(stats_dir, exist_ok=True)

        # Combine scores into one CSV file
        test_scores_df = pd.DataFrame(test_results)
        if len(test_scores_df) > 0:
            failed_df = pd.DataFrame(failed)
            test_scores_df = pd.concat([test_scores_df, failed_df])

        # Save all scores as CSV
        output_file = os.path.join(stats_dir, '1_all_scores.csv')
        test_scores_df.to_csv(output_file, index=False)

        # Scores per library across all presets and failed training counts
        if len(test_scores_df) > 0:
            summarized_scores = Utils.save_latex(test_scores_df, output_file.replace('csv', 'tex'))
            if plots and len(summarized_scores) > 2:
                Utils.save_heatmap(
                    summarized_scores,
                    os.path.join(stats_dir, 'heatmap.csv'),
                    os.path.join(stats_dir, 'heatmap.png'),
                )
                # Utils.plot_test_scores(test_scores, stats_dir, plots)

    @staticmethod
    def get_averaged_results(
        results_dir: str | Path,
        stats_dir: str | Path,
        test_results: list,
        dataset: str,
        failed: list,
    ) -> tuple:
        """Summarize results from CSV files

        :param str | Path results_dir: Path to results directory
        :param str | Path stats_dir: Path to statistics directory
        :param list test_results: List of test results
        :param str dataset: Dataset name
        :param list failed: List of failed training attempts
        :raises FileNotFoundError: If results file(s) are missing
        :return: Tuple of test results and failed training attempts
        """
        for library in os.listdir(results_dir):
            subdir = os.path.join(results_dir, library)
            for preset in os.listdir(subdir):
                preset_dir = os.path.join(subdir, preset)
                if subdir == str(stats_dir):
                    continue

                scores_path = os.path.join(preset_dir, f'{library}.csv')
                failed_path = os.path.join(preset_dir, 'failed.txt')

                if os.path.exists(scores_path):
                    df = pd.read_csv(scores_path, index_col=False)
                    test_results.append(
                        {
                            'library': library,
                            'preset': preset,
                            'file': dataset,
                            'failed': 0,
                            'num_iterations': len(df),
                            **df.mean(numeric_only=True).to_dict(),
                        }
                    )
                elif os.path.exists(failed_path):
                    failed.append(
                        {'library': library, 'preset': preset, 'file': dataset, 'failed': 1}
                    )
                else:
                    raise FileNotFoundError(f'Results file(s) missing in {preset_dir}')
        return test_results, failed

    @staticmethod
    def save_latex(df: pd.DataFrame, output_file: str) -> pd.DataFrame:
        """Save dataframe of results in a LaTeX file

        :param pd.DataFrame df: DataFrame of results
        :param str output_file: Path to .tex file
        """
        # Sort by GM-MAE-SR
        df = df.sort_values('GM-MAE-SR')

        # Filter columns and rename
        column_names = {
            'library': 'Library',
            'preset': 'Preset',
            'duration': 'Duration (sec.)',
            'GM-MAE-SR': 'GM-MAE-SRC',
            'MAE': 'MAE',
            'MASE': 'MASE',
            'MSE': 'MSE',
            'RMSE': 'RMSE',
            'Spearman Correlation': 'SRC',
        }
        df = df[column_names.keys()].rename(columns=column_names)

        # Format library names
        df['Library'] = df['Library'].str.capitalize()

        # Format presets
        df['Preset'] = (
            df['Preset']
            .str.replace('Fedot', 'FEDOT')
            .str.replace('Flaml', 'FLAML')
            .str.replace('Autokeras', 'AutoKeras')
            .str.replace('Autogluon', 'AutoGluon')
            .str.replace('preset-', '')
            .str.replace('proc-1', '')
            .str.replace('proc-10', '')
            .str.replace('-limit-3600', '')
            .str.replace('-limit-3240', '')
            .str.replace('-limit-3564', '')
            .str.replace('-limit-57', '')
            .str.replace('-limit-12', '')
            .str.replace('-limit-60', '')
            .str.replace('-limit-6', '')
            .str.replace('_', ' ')
            .str.capitalize()
            .str.replace(' ', '-')
            .str.replace('--', '-')
        )

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
    def plot_test_scores(
        test_scores: pd.DataFrame,
        stats_dir: str,
        plots: bool,
    ):
        """Plot test scores

        :param pd.DataFrame test_scores: Test scores
        :param str stats_dir: Path to output directory
        :param bool plots: If True, generate plots
        """
        test_scores['library'] = test_scores['library'].str.capitalize()

        # Ignore deprecated/unused presets
        test_scores = test_scores[~test_scores['preset'].isin(Utils.ignored_presets)]

        # Save overall scores and generate plots
        if plots and len(test_scores) > 2:
            # Bar plot of failed training attempts
            test_scores.plot.bar(y='failed', figsize=(35, 10))
            save_path = os.path.join(stats_dir, '3_failed_counts.png')
            Utils.save_plot('Failed Counts', save_path=save_path)

            # Box plots
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

        def plot_averages(group_col: str, cols_to_drop: list[str], fig_width: int, fig_height: int):
            df_grouped = test_scores.drop(cols_to_drop, axis=1).groupby(group_col)

            df_grouped.index = df_grouped[group_col]
            df_max = df_grouped.max()
            df_min = df_grouped.min()
            df_mean = df_grouped.mean()

            df_max.columns = [f'{c}_max' for c in df_max.columns.tolist()]
            df_min.columns = [f'{c}_min' for c in df_min.columns.tolist()]
            df_mean.columns = [f'{c}_mean' for c in df_mean.columns.tolist()]

            mean_scores = pd.concat([df_failed, df_max, df_min, df_mean], axis=1)

            output_file = os.path.join(stats_dir, f'3_mean_scores_by_{group_col}.csv')
            mean_scores.to_csv(output_file)

            if plots and len(mean_scores) > 2:
                # Bar plot of failed training attempts
                mean_scores.plot.bar(y='failed', figsize=(fig_width, fig_height))
                save_path = os.path.join(stats_dir, f'3_failed_counts_by_{group_col}.png')
                Utils.save_plot(
                    f'Failed Counts by {group_col}',
                    save_path=save_path,
                    legend=None,
                    yscale='linear',
                )

                # Box plots
                for col, filename, title in [
                    ('R2_mean', f'4_R2_mean_by_{group_col}.png', 'Mean R2'),
                    ('MAE_mean', f'5_MAE_mean_by_{group_col}.png', 'Mean MAE'),
                    ('MSE_mean', f'6_MSE_mean_by_{group_col}.png', 'Mean MSE'),
                    ('duration_mean', f'7_duration_mean_by_{group_col}.png', 'Mean Duration'),
                ]:
                    mean_scores.plot.bar(y=col, figsize=(fig_width, fig_height))
                    save_path = os.path.join(stats_dir, filename)
                    Utils.save_plot(title, save_path=save_path, legend=None, yscale='linear')

        # Plot mean scores by library
        plot_averages(
            group_col='library',
            cols_to_drop=['file', 'failed', 'preset'],
            fig_width=6,
            fig_height=3,
        )

        # # Plot mean scores by library/preset
        # test_scores['library-preset'] = test_scores['library'] + ': ' + test_scores['preset']
        # plot_averages(group_col='library-preset',
        #               cols_to_drop=['file', 'failed', 'preset', 'library'],
        #               fig_width=35,
        #               fig_height=10)

    @staticmethod
    def summarize_overall_results(results_dir: str, forecast_type: str, plots: bool = True):
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
                except pd.errors.EmptyDataError:
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
            summarized_scores = Utils.save_latex(
                all_scores, overall_scores_path.replace('csv', 'tex')
            )

            if plots and len(summarized_scores) > 2:
                logger.debug('Generating plots')
                Utils.save_heatmap(
                    summarized_scores,
                    os.path.join(stats_dir, 'metrics_corr_heatmap.csv'),
                    os.path.join(stats_dir, 'heatmap.png'),
                )
                Utils.plot_test_scores(all_scores, stats_dir, plots)
