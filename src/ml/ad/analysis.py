"""Anomaly detection analysis module for summarizing and plotting results of experiments."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

from ml.plots import Plotter


class Analysis:
    """Class for analyzing the results of anomaly detection experiments

    :param Path results_dir: The directory where the results of the experiments are saved.
    :param float test_set_size: The size of the test set as a fraction of the total dataset.
    :param str plots_dir_name: The name of the subdirectory to save plots in, defaults to 'plots'.
    :param str plots: The plotting behaviour, i.e. "all", "none", "skip_existing".
    """

    def __init__(
        self,
        results_dir: Path,
        test_set_size: float,
        plots: str = 'all',
        plots_dir_name: str = 'plots',
    ):
        self.results_dir = results_dir
        self.plots = plots
        self.test_set_size = test_set_size

        self.plots_dir = self.results_dir / plots_dir_name
        if not self.results_dir.exists():
            logger.warning(
                f'Results directory {self.results_dir} does not exist. Skipping analysis.'
            )

    def analyse_results(self, all_results: list[dict], all_metadata: dict) -> None:
        """Analyze the results of all experiments and save summary statistics and plots.

        :param list[dict] all_results: A list of dictionaries of results for each experiment.
        :param dict all_metadata: A dictionary containing the metadata for all experiments.
        """
        logger.debug(f'Saving metadata and results to: {self.results_dir}')

        # A. Save all metadata to a single JSON file and to CSV
        with open(self.results_dir / 'all_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=4)
        df_metadata = pd.DataFrame(all_metadata)
        # self._save_df_csv_and_tex(df_metadata, self.results_dir / 'all_metadata.csv')
        self._save_df_csv_and_tex(df_metadata.T, self.results_dir / 'all_metadata_T.csv')

        # B. Loop through each subdirectory and plot predictions vs true labels
        if self.plots not in ['none', None]:
            self._plot_predictions()

        if len(all_results) == 0:
            logger.info('No results to analyze. Exiting.')
            return

        # Filter ignored/unstable models from results
        df = pd.DataFrame(all_results)
        if 'param__model_name' in df.columns:
            df = df[~df['param__model_name'].isin(['pca', 'mcd', 'sod'])]

        # C. Save all results to a single CSV file
        self._save_df_csv_and_tex(df.copy(), self.results_dir / 'all_results.csv')

        # D. Save summary statistics of results to a separate CSV file
        df_info = df.describe(include='all').transpose()
        self._save_df_csv_and_tex(df_info, self.results_dir / 'all_results_info.csv', index=True)

        # E. Save latex table of results
        self._save_tables(df.copy())

        # F. Save plots of results
        if self.plots not in ['none', None]:
            self._save_results_plots(df.copy())

    def _save_tables(self, df: pd.DataFrame, min_f1_score: float = 0.5) -> None:
        """Save a LaTeX table of results to a .tex file in the results directory.

        :param pd.DataFrame df: The DataFrame containing the results to save as a LaTeX table.
        :param float min_f1_score: The minimum F1-score for rows to include, defaults to 0.5.
        """
        cols_tex = ['dataset'] + [
            col for col in df.columns if col.startswith('test__') or col.startswith('param__')
        ]

        if 'param__model_name' not in df.columns:
            logger.info(
                'param__model_name column not found in results. Skipping F1-score summary table.'
            )
            return

        # 1. TABLE: Mean, median, min, max, etc. of F1-score by dataset and model
        f1_summary = df.groupby(['dataset', 'param__model_name'])['test__f1_score'].agg(
            ['mean', 'median', 'min', 'max', 'std', 'count']
        )
        self._save_df_csv_and_tex(
            f1_summary.reset_index(), self.results_dir / 'f1_score_summary_by_dataset_and_model.csv'
        )

        # 2. TABLE: MEAN SCORES BY MODEL AND DATASET (regardless of F1-score)
        df_mean = df[cols_tex].rename(columns=self.readable)
        df_mean = df_mean.sort_values(by=['DATASET', 'MODEL NAME'])
        self._save_df_csv_and_tex(df_mean, self.results_dir / 'table_mean_unfiltered.csv')

        # 3. TEX FILE: LIST OF DROPPED MODELS (F1-SCORE BELOW min_f1_score)
        # Filter columns and rows where test__f1_score is greater than min_f1_score
        df_filtered = df[df['test__f1_score'] > min_f1_score]
        # Get names of models that were dropped entirely and save to tex file
        dropped_models = set(df['param__model_name']) - set(df_filtered['param__model_name'])
        with open(self.results_dir / 'dropped_models.tex', 'w', encoding='utf-8') as f:
            if dropped_models:
                f.write(
                    f'Excluded models with F1-score below {min_f1_score:.2f}: '
                    f'{", ".join(dropped_models)}.'
                )
            else:
                f.write(f'No models were dropped (all had F1-score > {min_f1_score:.2f}).')

        # 4. TABLE: ALL SCORES BY MODEL AND DATASET (FILTERED TO F1-SCORE > min_f1_score)
        # Make column names more readable
        df_filtered = df_filtered[cols_tex].rename(columns=self.readable)
        # Sort by dataset and model name
        df_filtered = df_filtered.sort_values(by=['DATASET', 'MODEL NAME'])
        # Save LaTeX table to file
        self._save_df_csv_and_tex(df_filtered, self.results_dir / 'table_all_scores_filtered.csv')

    def readable(self, col_name: str) -> str:
        """Convert column names to a more readable format for plot titles and labels."""
        return col_name.replace('param__', '').replace('test__', '').replace('_', ' ').upper()

    def _save_df_csv_and_tex(self, df: pd.DataFrame, csv_path: Path, index: bool = False) -> None:
        """Save a DataFrame to both CSV and LaTeX files in the results directory.

        :param pd.DataFrame df: The DataFrame to save.
        :param Path csv_path: Path to save the CSV file. The LaTeX file will use the same name.
        """
        # Readable dataset and model names for LaTeX table
        if 'dataset' in df.columns:
            df['dataset'] = df['dataset'].apply(self.readable)
        if 'model_name' in df.columns:
            df['model_name'] = df['model_name'].apply(self.readable)

        # Readable column names for LaTeX table
        df.columns = [self.readable(c) for c in df.columns]

        # Save to CSV
        df.to_csv(csv_path, index=index)
        # Save to LaTeX
        latex_table = df.to_latex(index=index, float_format='%.4f')
        with open(csv_path.with_suffix('.tex'), 'w', encoding='utf-8') as f:
            f.write(latex_table)

    def _save_results_plots(self, df: pd.DataFrame) -> None:
        """Save plots of results from a DataFrame to the results directory.

        :param pd.DataFrame df: The DataFrame containing the results to plot.
        """
        key_cols = ['dataset', 'param__model_name']
        test_score_cols = [col for col in df.columns if col.startswith('test__')]

        # 1. Create box plots of test scores by dataset and by model
        # Loop through each key column and create box plots for each test score column
        for key_col in key_cols:
            subdir = self.plots_dir / f'box_plots_by_{key_col}'
            for score_col in test_score_cols:
                self._generate_plot(subdir, 'box', df, score_col, key_col, ylim=(0.0, 1.0))

        # Calculate mean F1-score by model
        mean_f1_by_model = df.groupby('param__model_name')['test__f1_score'].mean()
        logger.trace(f'Mean F1-score by model:\n{mean_f1_by_model}')

        # 2. Create bar plot of mean F1-score by model (overall)
        # Plot by dataset again but only for the best model
        best_model = mean_f1_by_model.idxmax()
        df_best_model = df[df['param__model_name'] == best_model]
        subdir = self.plots_dir / 'box_plots_by_dataset-best_model'
        for score_col in test_score_cols:
            self._generate_plot(
                save_path=subdir,
                plot_type='bar',
                df=df_best_model,
                y_col=score_col,
                x_col='dataset',
                title=f'{score_col} by dataset for best model: {best_model}',
                ylim=(0.0, 1.0),
            )

    def _generate_plot(
        self,
        save_path: Path,
        plot_type: str,
        df: pd.DataFrame,
        y_col: str,
        x_col: str,
        title: str | None = None,
        figsize: tuple = (8, 4),
        v_lines: list[tuple[str, int]] | None = None,
        ylim: tuple | None = None,
    ) -> None:
        """Create and save a box plot of the specified score column by the specified key column.

        :param Path save_path: The file or directory to save the plot.
        :param str plot_type: The type of plot to create ('box', 'bar' or 'line').
        :param pd.DataFrame df: The DataFrame containing the results to plot.
        :param str y_col: The name of the column containing the scores to plot on the Y-axis.
        :param str x_col: The name of the column containing the categories to plot on the X-axis.
        :param str | None title: The title of the plot. If None, a default title will be generated.
        :param tuple figsize: The size of the figure to create, defaults to (8, 4).
        :param list[tuple[str, int]] | None v_lines: A list of (label, x-value) tuples at which to draw vertical lines.
        :param tuple | None ylim: The limits for the Y-axis, defaults to None.
        """
        save_path = Path(save_path)  # Ensure save_path is a Path object
        if save_path.is_dir():
            save_path = save_path / f'{plot_type}_{y_col}_by_{x_col}.svg'
        logger.trace(f'save_path: {save_path}')

        # Skip if plot already exists and we are in 'skip_existing' mode
        if self.plots == 'skip_existing' and save_path.exists():
            logger.trace('Plot exists, skipping...')
            return

        plt.figure(figsize=figsize)  # Set figure size

        # Create plot based on the specified type
        if plot_type == 'box':
            df.boxplot(column=y_col, by=x_col)
        elif plot_type == 'bar':
            df.groupby(x_col)[y_col].mean().plot(kind='bar')
        elif plot_type == 'line':
            df.plot()
        else:
            raise ValueError(f'Unknown plot type: {plot_type}')

        # Add vertical lines if specified
        if v_lines is not None:
            for label, x in v_lines:
                plt.axvline(x=x, color='red', linestyle='--')
                plt.text(x=x + 10, y=plt.ylim()[1] - 0.2, s=label, rotation=90, va='top')

        # Generate a readable title if not provided
        title = title or f'{plot_type} plot: {y_col} by {x_col}'

        # Save plot to file
        kwargs = {
            'title': self.readable(title),
            'xlabel': self.readable(x_col),
            'ylabel': self.readable(y_col),
            'xlabel_rotation': 90,
            'ylim': ylim,
        }
        Plotter.save_plot(save_path=save_path, **kwargs)
        # Plotter.save_plot(save_path=save_path.with_suffix('.png'), **kwargs)

    def _plot_predictions(self, skip_existing: bool = False) -> None:
        """Loop through each subdirectory and plot predictions vs true labels.

        :param bool skip_existing: Whether to skip plotting if the plot file already exists.
        """
        # Iterate through each dataset subdirectory (e.g. 'valve1__0')
        for dataset_subdir in self.results_dir.iterdir():
            if not dataset_subdir.is_dir():  # Ignore files
                continue

            # Iterate through each window subdirectory (e.g. 'window_50')
            for window_subdir in dataset_subdir.iterdir():

                # Expected predictions subdirectory:
                predictions_subdir = window_subdir / 'predictions'
                if not predictions_subdir.is_dir():  # Ignore files
                    # logger.warning(f'Predictions subdirectory not found: {predictions_subdir}')
                    continue

                # Iterate through CSV files (e.g. 'TimeSeriesODModel__model_name-IForest.csv')
                for prediction_file in predictions_subdir.iterdir():
                    if not prediction_file.is_file() or prediction_file.suffix != '.csv':
                        continue

                    # Load predictions and true labels
                    df_predictions = pd.read_csv(prediction_file, index_col=0)

                    # Skip if plot already exists and we are in 'skip_existing' mode
                    # plot_file = prediction_file.with_suffix('.svg')
                    plot_file = Path(
                        self.plots_dir,
                        'predictions',
                        dataset_subdir.name,
                        window_subdir.name,
                        f'{prediction_file.stem}.svg',
                    )
                    if skip_existing and plot_file.exists():
                        continue

                    # Calculate the index to draw the vertical line for the train/test split
                    v_line_index = int((1 - self.test_set_size) * len(df_predictions))

                    # Generate plot of predicted vs true labels
                    self._generate_plot(
                        save_path=plot_file,
                        plot_type='line',
                        df=df_predictions,
                        y_col='predicted',
                        x_col='index',
                        title=f'Predicted vs True Labels: {prediction_file.stem}',
                        figsize=(12, 4),
                        v_lines=[('Train/Test Split', v_line_index)],
                    )
