"""Analysis of TSB-AD results."""

import json
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as ss
import seaborn as sns

# Configure seaborn for better academic-style plots
sns.set_theme(style='whitegrid', context='paper', font_scale=1.2, font='serif', palette='Set2')

ALPHA = 0.01  # Significance level for statistical tests
CUSTOM_MODELS = ['LunarADModel', 'PyCaretADModel', 'TimeSeriesODModel']


class Analysis:
    """Class for analyzing TSB-AD results.

    :param Path results_dir: Directory to save results (default: 'results')
    :param list[Path] subdirs: List of results subdirectories
    :param Path plots_dir: Directory to save plots (default: 'results/plots')
    """

    # Results directory and subdirectories for TSB-AD-U and TSB-AD-M
    results_dir = Path('results')
    subdirs = [Path('results/TSB-AD-U'), Path('results/TSB-AD-M')]

    # Directory for saving plots
    plots_dir = results_dir / 'plots'

    # Expected metrics to be present in the CSV files
    available_metrics = [
        'AUC-PR',
        'AUC-ROC',
        'VUS-PR',
        'VUS-ROC',
        'Standard-F1',
        'PA-F1',
        'Event-based-F1',
        'R-based-F1',
        'Affiliation-F',
    ]

    def __init__(
        self,
        results_dir: Path = results_dir,
        subdirs: list[Path] = subdirs,
        plots_dir: Path = plots_dir,
    ):
        self.results_dir = results_dir
        self.subdirs = subdirs
        self.plots_dir = plots_dir

        self.results_dir.mkdir(parents=True, exist_ok=True)
        for subdir in self.subdirs:
            subdir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.custom_models = CUSTOM_MODELS + [self._readable_model_name(m) for m in CUSTOM_MODELS]

    def run_analysis(self) -> None:
        """Load results, aggregate metrics, rank models and plot results."""
        logger.info('Starting analysis of TSB-AD results...')
        # Read data from all subdirectories and concatenate into a single DataFrame
        df_all = self.load_results(self.subdirs)
        df_all['model_name'] = df_all['model_name'].apply(self._readable_model_name)

        df_all = df_all[~df_all['model_name'].isin(['DummyADModel'])]

        # Save aggregated results (mean, median, std, min, max) for each metric and model
        self.aggregate_results(df_all, rounding=2)
        df_ranks_mean = self.rank_models(df_all, rounding=2)

        kwargs_ = {'figsize': (7, 7), 'orient': 'h'}  # {'figsize': (10, 6), 'orient': 'v'}

        # Plot bar and box charts of mean ranks for each model
        logger.info('Generating plots of mean ranks for each model...')
        self.plot_bar_mean_ranks(df_ranks_mean, **kwargs_)
        kwargs_['figsize'] = (8, 8)
        self.sorted_box_plot(
            df=df_ranks_mean.reset_index(), x_col='Mean Rank', y_col='Model Name', **kwargs_
        )

        # Plot box plots of each metric by model, sorted by median metric value
        logger.info('Generating box plots for each metric by model...')
        for metric in self.available_metrics:
            self.sorted_box_plot(df=df_all, x_col=metric, y_col='Model Name', **kwargs_)

        # Perform statistical tests (Friedman and Conover) for the VUS-PR metric
        avg_ranks, test_results, groups = self.perform_statistical_tests(df_all, 'VUS-PR')

        # Plot the groups of models with no significant differences as a graph of nodes and edges
        self.plot_model_groups(groups)

        # Plot the critical difference diagram of average ranks with different significance levels
        kwargs_ = {
            'avg_ranks': avg_ranks,
            'test_results': test_results,
            'alpha': ALPHA,
            'figsize': (10, 5),
        }
        # Used text_h_margin=0.006 for unrotated image. This image is rotated 90 degrees in thesis.
        self.critical_difference_diagram(**kwargs_, groups=None, text_h_margin=0.01)
        self.critical_difference_diagram(**kwargs_, groups=groups, text_h_margin=0.01)

        # Plot correlation heatmap between metrics
        self.plot_correlation_heatmap(df_all)

        # Plot correlation between our results and the repository results
        df_tsb_scores, df_scores = self.compare_results(df_all, metric='VUS-PR')

        # Plot heatmap of differences between our results and the repository results
        self.plot_heatmap_differences(df_1=df_scores, df_2=df_tsb_scores, metric='VUS-PR')
        logger.success('Analysis completed successfully.')

    def load_results(self, subdirs: list[Path]) -> pd.DataFrame:
        """Load results from the specified subdirectories.

        :param list[Path] subdirs: List of subdirectories to load results from
        :return pd.DataFrame: Concatenated DataFrame containing all loaded results
        """
        dataframes = {}
        loaded_count = 0
        expected_count = 0

        for subdir in subdirs:
            if not subdir.exists():
                continue
            # Iterate through each folder in the subdirectory
            for folder in subdir.iterdir():
                if not folder.is_dir():
                    continue
                dataset_name = folder.name
                expected_count += 1

                # Check if the CSV file exists before trying to read it
                csv_path = folder / f'results_{dataset_name}.csv'
                if not csv_path.exists():
                    continue

                # Read the CSV file into a DataFrame and store it in the dictionary
                df = pd.read_csv(folder / f'results_{dataset_name}.csv')
                dataframes[dataset_name] = df
                loaded_count += 1

        logger.info(f'Loaded {loaded_count} of {expected_count} expected dataframes.')
        df_all = pd.concat(dataframes.values(), ignore_index=True)
        # df_all.to_csv('df_all.csv', index=False)
        return df_all

    def aggregate_results(
        self, df_all: pd.DataFrame, subdir_name: str = 'aggregated', rounding: int = 2
    ) -> None:
        """Save the mean, median, std, min, and max of each metric for each model.

        :param pd.DataFrame df_all: DataFrame containing all results with columns 'dataset', 'Model Name', and metrics
        :param str subdir_name: Name of the subdirectory to save the aggregated results (default: 'aggregated')
        :param int rounding: Number of decimal places to round the aggregated values (default: 2)
        """
        df = df_all.copy().rename(columns={'model_name': 'Model Name'})
        grouped = df.groupby('Model Name')[self.available_metrics]
        aggregations = {
            'mean': grouped.mean().round(rounding),
            'median': grouped.median().round(rounding),
            'std': grouped.std().round(rounding),
            'min': grouped.min().round(rounding),
            'max': grouped.max().round(rounding),
        }
        for agg_name, df_agg in aggregations.items():
            Path(self.results_dir / subdir_name).mkdir(parents=True, exist_ok=True)
            df_agg.to_csv(self.results_dir / subdir_name / f'{agg_name}_metrics.csv')
            df_agg.to_latex(
                self.results_dir / subdir_name / f'{agg_name}_metrics.tex',
                float_format=f'%.{rounding}f',
                caption=f'{agg_name.capitalize()} metrics for all models.',
                label=f'tab:ch5:{agg_name}_metrics',
                bold_rows=True,
                column_format='l' + 'r' * len(self.available_metrics) + 'r',
                escape=True,
            )

    def rank_models(self, df_all: pd.DataFrame, rounding: int = 2) -> pd.DataFrame:
        """Get the average rank of each model across all metrics.

        :param pd.DataFrame df_all: DataFrame with columns 'dataset', 'model_name', and metrics
        :param int rounding: Number of decimal places to round the rank values (default: 2)
        :return pd.DataFrame: DataFrame with the average ranks for each model
        """
        # Calculate the average rank of each model across all metrics (lower is better)
        df_ = df_all.groupby('model_name')[self.available_metrics].mean().rank(ascending=False)
        df_.index.name = 'Model Name'
        df_['Mean Rank'] = df_.mean(axis=1).round(rounding)
        # Move Mean Rank column to 1st position
        cols = df_.columns.tolist()
        cols.insert(0, cols.pop(cols.index('Mean Rank')))
        # Save to file
        df_.to_csv(self.results_dir / 'ranks_mean.csv')
        df_.to_latex(
            self.results_dir / 'ranks_mean.tex',
            float_format=f'%.{rounding}f',
            caption='Average ranks of models across all metrics (lower is better).',
            label='tab:ch5:mean_ranks_all_metrics',
            bold_rows=True,
            column_format='l' + 'r' * len(self.available_metrics) + 'r',
            escape=True,
        )
        return df_

    def sorted_box_plot(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        figsize: tuple[int, int] = (10, 6),
        orient: str = 'v',
    ) -> None:
        """Draw a box plot of the specified metric by model, sorted by median metric value.

        :param pd.DataFrame df: DataFrame of all results with columns including x_col and y_col
        :param str x_col: Column name to use for the x-axis (e.g., 'Model Name')
        :param str y_col: Column name to use for the y-axis (e.g., 'AUC-PR')
        :param tuple[int, int] figsize: Figure size for the plot
        :param str orient: Orientation of the box plot ('v' for vertical, 'h' for horizontal)
        """
        # Set labels and axes based on orientation
        if orient == 'v' and x_col == 'Model Name':
            metric_col, x_label, y_label = y_col, 'Model', y_col
        elif orient == 'h' and y_col == 'Model Name':
            metric_col, x_label, y_label = x_col, x_col, 'Model'
        else:
            raise ValueError(
                'Invalid combination of x_col, y_col, and orient. For vertical orientation, '
                'x_col must be "Model Name". For horizontal orientation, y_col must be '
                f'"Model Name". Received x_col: {x_col}, y_col: {y_col}, orient: {orient}.'
            )

        # Sort by median score for each model
        df_ = df.copy().rename(columns={'model_name': 'Model Name'})
        df_['Model Name'] = df_['Model Name'].astype(str)
        model_order = (
            df_.groupby('Model Name')[metric_col].median().sort_values(ascending=False).index
        )
        df_['Model Name'] = pd.Categorical(df_['Model Name'], categories=model_order, ordered=True)

        # Box plot of scores by model sorted by median score
        plt.figure(figsize=figsize)
        kwargs_ = {
            'hue': 'Model Name',
            'data': df_,
            'orient': orient,
            'palette': [
                'red' if model in self.custom_models else 'lightblue' for model in model_order
            ],
        }
        if orient == 'h':
            sns.boxplot(x=metric_col, y='Model Name', **kwargs_)
        else:
            sns.boxplot(x='Model Name', y=metric_col, **kwargs_)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if orient == 'v':
            plt.xticks(rotation=90)
        plt.tight_layout()
        # Save the plot
        self._save_plot(self.plots_dir / 'box_all_models' / f'box_{orient}_{metric_col}.png')

    def plot_bar_mean_ranks(
        self, df_ranks_mean: pd.DataFrame, figsize: tuple[int, int] = (10, 6), orient: str = 'v'
    ):
        """Plot a bar chart of the mean ranks of each model.

        :param pd.DataFrame df_ranks_mean: DataFrame with the average ranks for each model.
        :param tuple[int, int] figsize: Figure size for the plot.
        :param str orient: Orientation of the bar plot ('v' for vertical, 'h' for horizontal).
        """
        df_ = df_ranks_mean.sort_values('Mean Rank', ascending=True)  # Sort by mean rank
        plt.figure(figsize=figsize)

        # Bar plot of mean ranks for each model
        kwargs = {
            'data': df_,
            'orient': orient,
            'linewidth': 1,
            'edgecolor': 'gray',
            'palette': [
                'teal' if model in self.custom_models else 'lightgray' for model in df_.index
            ],
        }
        if orient == 'h':
            x_label, y_label = 'Mean Rank', 'Model'
            sns.barplot(x=x_label, y=df_.index, hue='Model Name', **kwargs)
        elif orient == 'v':
            x_label, y_label = 'Model', 'Mean Rank'
            sns.barplot(x=df_.index, y=y_label, hue='Model Name', **kwargs)
        else:
            raise ValueError(f'Invalid orientation: {orient}. Use "h" or "v".')

        # plt.title('Mean Rank of Each Model Across All Metrics')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        if orient == 'v':
            plt.xticks(rotation=90)
        plt.tight_layout()
        self._save_plot(self.plots_dir / f'bar_{orient}_mean_rank.svg')

    def perform_statistical_tests(
        self, df_all: pd.DataFrame, metric: str = 'VUS-PR'
    ) -> tuple[pd.Series, pd.DataFrame, list[set[str]]]:
        """Perform the Friedman rank sum test and Conover post-hoc test.

        :param pd.DataFrame df_all: DataFrame with columns 'dataset', 'model_name', and metrics
        :param str metric: Name of the metric to perform tests on
        :return tuple[pd.Series, pd.DataFrame, list[set[str]]]: Average ranks, Conover test results, and model groups
        """
        df_scores: pd.DataFrame = df_all[['dataset', 'model_name', metric]].copy()
        avg_ranks: pd.Series = (
            df_scores.groupby('dataset')[metric]
            .rank(pct=True)
            .groupby(df_scores['model_name'])
            .mean()
        )
        # Drop rows with NaN values in the metric column (if any) before performing statistical tests
        df_scores = df_scores.dropna(subset=[metric])
        # Perform Friedman rank sum test to determine if there are significant differences between models
        self.friedman_rank_sum_test(df_scores, score_col=metric, group_col='model_name')
        test_results, groups = self.conover_posthoc_test(
            df_scores, score_col=metric, group_col='model_name', block_col='dataset'
        )
        return avg_ranks, test_results, groups

    def friedman_rank_sum_test(
        self, df: pd.DataFrame, score_col: str, group_col: str = 'model_name'
    ) -> None:
        """Perform the Friedman rank sum test to determine if there are significant differences between models.

        :param pd.DataFrame df: Input DataFrame
        :param str score_col: Name of the column containing the scores to be tested
        :param str group_col: Name of the column containing the grouping variable (e.g., 'model_name')
        """
        # Friedman Rank Sum Test
        # - Null hypothesis: all models come from same distribution -> stop
        # - Alternative hypothesis: at least one model comes from a different distribution -> proceed
        scores_by_model = df.groupby(group_col)[score_col].apply(list)
        try:
            friedman_result = ss.friedmanchisquare(*scores_by_model.values)
            if friedman_result.pvalue < ALPHA:
                logger.debug(
                    f'\nSignificant differences found between models (p < {ALPHA}): {friedman_result}\n'
                )
            else:
                logger.debug(
                    f'\nNo significant differences found between models (p >= {ALPHA}): {friedman_result}\n'
                )
        except ValueError as e:
            logger.error(f'Error performing Friedman test: {e}')
            # logger.error(f'Data:\n{scores_by_model}\n')
            # scores_by_model.to_csv('friedman_test_error.csv')
            # for model, scores in scores_by_model.items():
            #     logger.error(f'Model: {model}, Number of scores: {len(scores)}')
            # exit()

    def conover_posthoc_test(
        self, df: pd.DataFrame, score_col: str, group_col: str, block_col: str
    ) -> tuple[pd.DataFrame, list[set[str]]]:
        """Perform the Conover post-hoc test to determine which models are significantly different.

        :param pd.DataFrame df: DataFrame containing 'dataset', 'model_name', and the metric
        :param str score_col: Name of the column containing the scores to be tested
        :param str group_col: Name of the column containing the grouping variable (e.g., 'model_name')
        :param str block_col: Name of the column containing the blocking variable (e.g., 'dataset')
        :return tuple[pd.DataFrame, list[set[str]]]: DataFrame (pairwise p-values) and list of model groups
        """
        # ### Post-Hoc Conover Test
        # - Null hypothesis: the two models come from the same distribution -> stop
        # - Alternative hypothesis: the two models come from different distributions -> proceed
        test_results = sp.posthoc_conover_friedman(
            df,
            melted=True,
            block_col=block_col,
            block_id_col=block_col,
            group_col=group_col,
            y_col=score_col,
        )
        if (test_results < ALPHA).any().any():
            logger.debug(f'Significant pairwise differences found between models (p < {ALPHA})\n')
        else:
            logger.debug(
                f'No significant pairwise differences found between models (p >= {ALPHA})\n'
            )
        # sp.sign_plot(test_results)

        # Group models that are not significantly different from each other based on the Conover test results
        groups: list[set[str]] = []
        for (model1, model2), is_insignificant in (test_results >= ALPHA).stack().items():
            if is_insignificant and model1 != model2:
                # Find the group this model belongs to, or create a new one
                group = None
                for g in groups:
                    if model1 in g:
                        group = g
                        break
                if group is None:
                    group = set()
                    groups.append(group)
                group.add(model1)
                group.add(model2)

        results_path = self.results_dir / f'model-groups_alpha-{ALPHA}.txt'
        with open(results_path, 'w', encoding='utf-8') as f:
            # Record the groups of models with no significant differences
            logger.debug(f'Model groups with no significant differences (p >= {ALPHA}):\n')
            f.write(f'Model groups with no significant differences (p >= {ALPHA}):\n')
            for i, group in enumerate(groups, 1):
                logger.debug(f'Group {i}: {", ".join(sorted(group))}')
                f.write(f'Group {i}: {", ".join(sorted(group))}\n')

        # Save groups to a JSON file
        with open(results_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump([sorted(group) for group in groups], f)

        return test_results, groups

    def plot_model_groups(self, groups: list[set[str]]) -> None:
        """Plot the groups of models with no significant differences as a graph of nodes and edges.

        :param list[set[str]] groups: List of sets of model names, where each set contains models that are not significantly different from each other
        """
        edge_colors = []
        for group in groups:
            hue: list = sns.color_palette('Set2', n_colors=len(groups))[groups.index(group)]
            edge_colors.extend(
                [hue] * (len(group) * (len(group) - 1) // 2)
            )  # Number of edges in a complete graph of size len(group))

        # Connect models with edges if they are in the same group (i.e., not significantly different)
        G = nx.Graph()
        for group in groups:
            for model in group:
                G.add_node(model)
            for model1 in group:
                for model2 in group:
                    if model1 != model2:
                        G.add_edge(model1, model2)
        # Draw graph
        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G, k=0.5, iterations=3, seed=1)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color='lightblue',
            edge_color=edge_colors,
            node_size=3500,
            font_size=9,
            font_family='serif',  # font_weight='bold'
        )
        # plt.tight_layout() # "UserWarning: This figure includes Axes that are not ..."
        self._save_plot(self.plots_dir / f'model_groups_alpha-{ALPHA}.svg')

    def critical_difference_diagram(
        self,
        avg_ranks: pd.Series,
        test_results: pd.DataFrame,
        alpha: float,
        figsize: tuple[int, int] = (10, 5),
        groups: list[set[str]] | None = None,
        **kwargs,
    ) -> None:
        """Critical difference diagram of average ranks with different significance levels

        :param pd.Series avg_ranks: Series with model names as index and average ranks as values
        :param pd.DataFrame test_results: DataFrame with significance results
        :param float alpha: Significance level
        :param tuple[int, int] figsize: Figure size for the diagram
        :param list[set[str]] | None groups: Sets of models that are not significantly different
        """
        plt.figure(figsize=figsize, dpi=300)

        # Crossbar = lines connecting model groups with no significant differences
        crossbar_props = {'marker': 'D', 'markersize': 3, 'linewidth': 2.0, 'color': 'grey'}
        # Elbow = lines connecting each model to its rank on the diagram
        elbow_props = {'linewidth': 0.5, 'color': 'black'}

        # Get hue from groups
        if groups is None:
            hue = None
        else:
            del crossbar_props['color']
            del elbow_props['color']

            # Some palettes: "Set2", "tab10", "colorblind", "rocket", "dark", "hls"
            palette = sns.color_palette('icefire', n_colors=len(groups) + 4)
            hue = {}
            for group in groups:
                color_index = groups.index(group) + 2
                for model_name in group:
                    hue[model_name] = palette[color_index]

            # Assign a default color to any model not in any group
            for model_name in avg_ranks.index:
                if model_name not in hue:
                    hue[model_name] = 'black'  # sns.color_palette('Set2')[-1]

        sp.critical_difference_diagram(
            ranks=avg_ranks,
            sig_matrix=test_results,
            alpha=alpha,
            # hue=hue,
            color_palette=hue,
            crossbar_props=crossbar_props,
            elbow_props=elbow_props,
            label_fmt_left='{label} [{rank:.2f}]  ',
            label_fmt_right='  [{rank:.2f}] {label}',
            text_h_margin=kwargs.get('text_h_margin', 0.006),
        )
        plt.tight_layout()
        color = 'uncolored' if groups is None else 'colored'
        self._save_plot(self.plots_dir / f'critical-diff_{color}_alpha-{alpha}.svg')

    def plot_correlation_heatmap(
        self, df_all: pd.DataFrame, figsize: tuple[int, int] = (6, 6)
    ) -> None:
        """Plot a heatmap of the Spearman rank correlation between the metrics.

        :param pd.DataFrame df_all: DataFrame containing all results with columns 'dataset', 'model_name', and metrics
        :param tuple[int, int] figsize: Figure size for the heatmap
        """
        correlation_matrix = df_all[self.available_metrics].corr(method='spearman').round(2)
        correlation_matrix.to_csv(self.results_dir / 'metrics_spearman_correlation.csv')
        # Visualise the correlation matrix as a heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(
            correlation_matrix, annot=True, vmin=0, vmax=1, cbar=False, square=True, cmap='Blues'
        )
        plt.tight_layout()
        self._save_plot(self.plots_dir / 'spearman_heatmap_metrics.svg')

    def compare_results(
        self, df_all: pd.DataFrame, metric: str = 'VUS-PR'
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Plot a scatter plot of the correlation between two metrics.

        :param pd.DataFrame df_all: DataFrame containing all results with columns 'dataset', 'model_name', and metrics
        :param str metric: The metric for which to plot the correlation with the repository results
        :return tuple[pd.DataFrame, pd.DataFrame]: DataFrames of the repository scores and our scores
        """
        df_tsb_scores = pd.read_csv(
            f'benchmark_exp/benchmark_eval_results/uni_mergedTable_{metric}.csv'
        )  # Read the repository results for the specified metric

        # Extract dataset names from the file column and create a new dataset column
        df_tsb_scores['dataset'] = df_tsb_scores['file'].apply(lambda x: Path(x).stem)
        df_tsb_scores = df_tsb_scores.drop(columns=['file'])

        # Filter to only include datasets that are in our results
        common_datasets = set(df_tsb_scores['dataset']).intersection(set(df_all['dataset']))
        df_tsb_scores = df_tsb_scores[df_tsb_scores['dataset'].isin(common_datasets)]
        # logger.debug(f'Common datasets between our results and the repository: {len(common_datasets)}')

        # Filter other columns
        df_tsb_scores = df_tsb_scores.drop(
            columns=[
                'ts_len',
                'anomaly_len',
                'num_anomaly',
                'avg_anomaly_len',
                'anomaly_ratio',
                'point_anomaly',
                'seq_anomaly',
            ]
        )

        # df_tsb_scores columns are model names and rows are datasets.
        # Reformat df_all to have the same format for the specified metric.
        df_scores = df_all[['dataset', 'model_name', metric]]
        df_scores = df_scores.groupby(['dataset', 'model_name'])[metric].mean().reset_index()
        df_scores = df_scores.pivot(index='dataset', columns='model_name', values=metric)

        # Sort columns alphabetically
        df_tsb_scores = df_tsb_scores.sort_index(axis=1)
        df_scores = df_scores.sort_index(axis=1)

        # Place dataset column first
        df_tsb_scores = df_tsb_scores[
            ['dataset'] + [col for col in df_tsb_scores.columns if col != 'dataset']
        ]
        df_scores = df_scores.reset_index().rename(columns={'dataset': 'dataset'})

        # Standardize column names to ensure they match between the two DataFrames
        df_tsb_scores = self._standardize_column_names(df_tsb_scores)
        df_scores = self._standardize_column_names(df_scores)

        # Identify the models that are present in both DataFrames (excluding the dataset column)
        models_in_common = set(df_tsb_scores.columns).intersection(set(df_scores.columns)) - {
            'dataset'
        }

        # Filter both DataFrames to only include the models in common
        df_tsb_scores = df_tsb_scores[['dataset'] + sorted(models_in_common)]
        df_scores = df_scores[['dataset'] + sorted(models_in_common)]

        # Filter both DataFrames to only include the datasets in common
        common_datasets = set(df_tsb_scores['dataset']).intersection(set(df_scores['dataset']))
        df_tsb_scores = df_tsb_scores[df_tsb_scores['dataset'].isin(common_datasets)].reset_index(
            drop=True
        )
        df_scores = df_scores[df_scores['dataset'].isin(common_datasets)].reset_index(drop=True)
        # logger.debug(
        #     'After filtering to common models and datasets:\ndf_tsb_scores.shape: '
        #     f'{df_tsb_scores.shape}, df_scores.shape: {df_scores.shape}\n'
        # )

        # Set the dataset column as the index for both DataFrames to prepare for comparison
        df_tsb_scores = df_tsb_scores.set_index('dataset', drop=True)
        df_scores = df_scores.set_index('dataset', drop=True)

        return df_tsb_scores, df_scores

    def plot_heatmap_differences(
        self,
        df_1: pd.DataFrame,
        df_2: pd.DataFrame,
        metric: str,
        figsize: tuple[int, int] = (17, 30),
    ):
        """Plot a heatmap of the differences between two DataFrames.

        :param pd.DataFrame df_1: First DataFrame (e.g., VUS-PR results)
        :param pd.DataFrame df_2: Second DataFrame (e.g., repository results)
        :param str metric: The metric for which to plot differences
        :param tuple[int, int] figsize: Figure size for the heatmap
        """
        differences = (df_1 - df_2).dropna(how='all')
        differences.to_csv(self.results_dir / f'differences_{metric}.csv')

        plt.figure(figsize=figsize, dpi=300)
        sns.heatmap(
            differences, annot=True, cmap='coolwarm', center=0, square=True, cbar=False, fmt='.2f'
        )
        plt.tight_layout()
        self._save_plot(self.plots_dir / f'heatmap_differences_{metric}.png')

    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names.

        :param pd.DataFrame df: DataFrame containing columns to be standardized
        :return pd.DataFrame: DataFrame with standardized column names
        """
        df = df.copy()
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(' ', '_')
            .str.replace('-', '_')
            .str.replace('(', '')
            .str.replace(')', '')
        )
        return df

    def _readable_model_name(self, model_name: str) -> str:
        """Convert model names to a more readable format.

        :param str model_name: Original model name to be converted
        :return str: Readable model name
        """
        model_name = (
            str(model_name)
            .strip()
            .replace('_', '-')
            .replace('ADModel', '')
            .replace('AD', '')
            .replace('TimeSeriesODModel', 'TSOD')
        )
        return model_name

    def _save_plot(self, path: Path | str):
        """Save the current plot to the specified path in both PNG and SVG formats.

        :param Path | str path: The file path where the plot should be saved as PNG and SVG.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        logger.debug(path)
        plt.savefig(path.with_suffix('.png'), dpi=300)
        plt.savefig(path.with_suffix('.svg'))


if __name__ == '__main__':
    Analysis().run_analysis()
