"""Functionality for reporting the results of source code analysis."""

import json
import sys
import textwrap
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns

from ml import LIBRARIES
from ml.logs import logger

# LaTeX formatting
BOLD = '\\textbf'
ITALIC = '\\textit'
BORDER = '\\hline'


class Reporting:
    """SCA reporting functionality."""

    FOOTNOTES = {
        'CC': ITALIC + '{ CC: Cyclomatic Complexity}',
        'LLoC': ITALIC + '{ LLoC: Logical Lines of Code}',
        'LoC': ITALIC + '{ LoC: Lines of Code}',
        ' MI': ITALIC + '{ MI: Maintainability Index}',
        'NCLoC': ITALIC + '{ NCLoC: Non-Comment Lines of Code}',
        'SLoC': ITALIC + '{ SLoC: Source Lines of Code}',
        'SQALE': ITALIC + '{ SQALE: Software Quality Assessment based on Lifecycle Expectations}',
    }
    INTEGER_PATTERNS = [  # Column names patterns to convert to integers
        'Blocks',
        'Branches',
        'Bugs',
        'Cognitive Complexity',
        'Comments',
        'Conditions',
        'Classes',
        'Duplicated',
        'Development Cost',
        'Files',
        'Functions',
        'Issues',
        'Lines',
        'Num.',
        'oC',
        'Reliability',
        'Remediation Effort',
        'Sonar: Complexity',
        'Smells',
        'spots',
        'SQALE Index',
        'Statements',
        'Total CC',
        'Violations',
    ]
    LONGEST_NAME = max(len(name) for name in LIBRARIES.values())
    # fmt: off
    RANK_METRICS = [
        'bandit__Total Issues', 'coverage__line-rate', 'coverage__branch-rate',
        'pylint__score', 'ruff__Score', 'Prospector Num. Issues',
        # ##### Radon #####
        'radon-cc__mean-cc_class', 'radon-cc__total-cc_class',  # Complexity
        'radon-cc__mean-cc_function', 'radon-cc__total-cc_function',
        'radon-cc__mean-cc_method', 'radon-cc__total-cc_method',
        'radon-hal__Mean Halstead Bugs', 'radon-hal__Total Halstead Bugs',  # Halstead
        'radon-hal__Mean Halstead Effort', 'radon-hal__Total Halstead Effort',
        'radon-hal__Mean Halstead Difficulty', 'radon-hal__Total Halstead Difficulty',
        'radon-hal__Mean Halstead Volume', 'radon-hal__Total Halstead Volume',
        'radon-mi__Mean Maintainability Index',  # Maintainability
        # ##### SonarQube #####
        'sonar__bugs',
        'sonar__code_smells',
        'sonar__cognitive_complexity',
        'sonar__complexity',
        'sonar__line rate',  # 'sonar__coverage',
        'sonar__development_cost',
        'sonar__duplicated_blocks',
        'sonar__duplicated_files',
        'sonar__duplicated_lines',
        'sonar__duplicated_lines_density',
        'sonar__file_complexity',
        'sonar__reliability_remediation_effort',
        'sonar__security_hotspots',
        'sonar__sqale_debt_ratio',
        'sonar__sqale_index',
        'sonar__uncovered_lines',
        'sonar__violations',
    ]
    # fmt: on
    SORT_COLUMNS = {
        'Complexity': ('sonar__complexity', True),
        'Test Coverage': ('coverage__line-rate', False),
        'Duplication': ('Sonar (Duplications)__lines', True),
        'Linting': ('sonar__violations', True),
        'Maintainability and Reliability': ('sonar__sqale_index', True),
        'Program Size': ('radon-hal__Total Halstead Volume', True),
    }
    SUMMARY_DIR = 'SUMMARY'

    @classmethod
    def save_results(
        cls, df: pd.DataFrame, output_dir: str | Path, ignored_keys_by_tool: dict[str, list[str]]
    ) -> pd.DataFrame:
        """Save the results of the SCA analysis to a file.

        :param pd.DataFrame df: The results as a pandas DataFrame.
        :param str | Path output_dir: Path to save the analysis
        :param dict[str, list[str]] ignored_keys_by_tool: Ignored keys by tool.
        :return pd.DataFrame df_results: The results as a pandas DataFrame.
        """
        output_dir = Path(output_dir, cls.SUMMARY_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        Path(output_dir, 'csv').mkdir(parents=True, exist_ok=True)
        Path(output_dir, 'tables').mkdir(parents=True, exist_ok=True)
        Path(output_dir, 'tex').mkdir(parents=True, exist_ok=True)

        # Ensure correct library names
        df['name'].replace(LIBRARIES, inplace=True)

        # Save all results to a CSV file
        results_file = Path(output_dir, 'csv', 'results.csv')
        df.to_csv(results_file, index=False)
        logger.info(f'Saved results to {results_file}')

        # Filter, combine and/or format columns
        dropped_cols, df_summary, keys_by_tool = cls.filter_rename_columns(df, ignored_keys_by_tool)

        # Summarize results and export to CSV, Markdown, and LaTeX
        used_cols, _, median_ranks = cls.summarize_by_category(df_summary, keys_by_tool, output_dir)
        unused_cols = [col for col in df_summary.columns if col not in used_cols]

        # Save metadata to a JSON file
        metadata = {
            'metrics_dropped': dropped_cols,
            'metrics_ignored': ignored_keys_by_tool,
            'metrics_used': used_cols,
            'metrics_unused': unused_cols,
            'num_repos': len(df_summary),
        }
        metadata_file = Path(output_dir, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f'Saved metadata to {metadata_file}')

        # Plots
        logger.debug('Saving plots...')
        df_ranks = cls.create_plots(df_summary, output_dir)

        # Save median ranks
        logger.debug('Saving median ranks by category...')
        df_by_category = pd.DataFrame(median_ranks)
        df_ranks_summary = df_ranks[[col for col in df_ranks.columns if 'Rank' in col]]
        df_ranks_summary = pd.concat([df_ranks_summary, df_by_category], axis=1)

        # Sort by median rank and save to CSV
        df_ranks_summary.index.name = 'Library'
        df_ranks_summary.sort_values(by='Median Rank', inplace=True)
        df_ranks_summary.to_csv(Path(output_dir, 'csv', 'summary_ranks.csv'), index=True)
        df_ranks_summary.T.to_csv(Path(output_dir, 'csv', 'T_summary_ranks.csv'), index=True)

        # Plot heatmap
        renamed_ = {
            'Test Coverage': 'Coverage',
            'Maintainability and Reliability': 'Maintainability',
        }
        df_ranks_summary.rename(columns=renamed_, inplace=True)
        cls.plot_libraries(df_ranks_summary, 'heatmap', 'Median Ranks', output_dir, (10, 7), 8)
        cls.plot_libraries(df_ranks_summary, 'spearman', 'Median Ranks', output_dir, (10, 10), 8)

        # Correlation heatmap with sortable metrics only
        df_sortable = df_ranks[cls.RANK_METRICS]
        df_sortable.columns = [cls.format_col_name(c) for c in df_sortable.columns.tolist()]
        corr_kwargs = {
            'data': df_sortable,
            'kind': 'spearman',
            'output_dir': output_dir,
            'figsize': (20, 15),
            'fontsize': 8,
            'show_tool': True,
            'compact': False,
            'show_p_values': True,
        }
        for threshold in [0.05, 0.01]:
            corr_kwargs['p_value_threshold'] = threshold
            corr_kwargs['metric'] = f'Spearman correlation ({threshold})'
            cls.plot_libraries(**corr_kwargs)

        # Save as LaTeX
        column_format = 'l|'
        for col in df_ranks_summary.columns:
            column_format += cls._column_width(df_ranks_summary[col], col, extra_width=0.1)

        df_ranks_summary.rename(columns={'Median Rank': '\\textbf{Median Rank}'}, inplace=True)

        path = Path(output_dir, 'tables', 'summary_ranks.tex')
        df_ranks_summary.astype(int).style.to_latex(
            path,
            column_format=column_format,
            caption='Median ranks of AutoML libraries by category',
            label='tab:r:summary_ranks',
            hrules=True,
            multicol_align='c',
            position='!htbp' if len(df.columns) <= 8 else '',
            position_float='centering',
        )
        cls.format_tex_table(df_ranks_summary, path, sort_index=None, text_size='\\footnotesize\n')

        cls.save_notes_as_tex(df_summary, df_ranks_summary, output_dir)
        return df_summary

    @classmethod
    def filter_rename_columns(cls, df: pd.DataFrame, ignored_keys_by_tool: dict[str, list[str]]):
        """Filter and rename the columns of the results dataframe for analysis.

        :param pd.DataFrame df: The results as a pandas DataFrame.
        :param dict[str, list[str]] ignored_keys_by_tool: Ignored keys by tool.
        :return tuple: Dropped columns, formatted DataFrame, and grouped metrics.
        """
        df_summary = df[sorted(df.columns)].copy()  # Sort columns
        # Capitalize library name to prevent sorting error...
        df_summary['name'] = df_summary['name'].apply(lambda x: x.upper()[0] + x[1:])
        df_summary.set_index('name', drop=True, inplace=True)  # Set library name as index

        # Drop columns with only one unique value
        counts = df_summary.nunique()
        dropped = [col for col in counts.index.to_list() if counts[col] < 2]
        if len(dropped) > 0:
            logger.warning(f'Dropping columns with only one unique value: {dropped}')

        # Drop ignored columns
        dropped += ['path', 'git__LoC']
        for keys in ignored_keys_by_tool.values():
            dropped += keys
        df_summary.drop(columns=dropped, inplace=True, errors='ignore')

        # Group metric keys by tool
        keys_by_tool: dict[str, list[str]] = {}
        for col in df_summary.columns:
            if '__' in col:
                tool = col.split('__')[0]
                keys_by_tool[tool] = keys_by_tool.get(tool, []) + [col]
            elif col not in ['name', 'path']:
                raise ValueError(f'Invalid column name "{col}"')

        # Bandit: score = sum of all bandit issues
        metric_keys = [
            key
            for key in keys_by_tool['bandit']
            if key.startswith('bandit__B') and key not in ['bandit__B101', 'bandit__B311']
        ]  # B101: assertion, B311: use of random
        df_summary['bandit__Total Issues'] = df_summary[metric_keys].sum(axis=1)
        # df_summary['bandit__Issues per Line'] = df_summary['bandit__Total Issues'] / df_summary['bandit__loc']
        dropped += keys_by_tool['bandit']

        # Coverage: difference between Sonar and Coverage.py
        df_summary['Difference__(\\%)'] = df_summary['sonar__coverage'] - (
            df_summary['coverage__line-rate'] * 100
        )
        keys_by_tool['Difference'] = ['Difference__(\\%)']

        # Prospector: Rename columns
        renamed = [k.replace('prospector__', '') for k in keys_by_tool['prospector']]
        df_summary.rename(columns=dict(zip(keys_by_tool['prospector'], renamed)), inplace=True)
        keys_by_tool['prospector'] = renamed
        # Prospector: Drop non-score columns
        dropped += [k for k in renamed if not k.endswith(' Issues')]
        # Prospector: replace NaNs with 0
        df_summary[keys_by_tool['prospector']].fillna(0, inplace=True)

        # Pylint: only keep score
        dropped += [k for k in keys_by_tool['pylint'] if k != 'pylint__score']

        # Ruff: only keep score
        df_summary['ruff__Score'] = df_summary['ruff__ruff Score']
        dropped += keys_by_tool['ruff']

        # Sonar: rename Coverage to 'Line Rate' for consistency with Coverage.py
        df_summary['sonar__line rate'] = df_summary['sonar__coverage'] / 100
        del df_summary['sonar__coverage']
        keys_by_tool['sonar'].remove('sonar__coverage')
        keys_by_tool['sonar'].append('sonar__line rate')

        # Sonar: drop 'new' and 'issues' metrics
        for term in ['_new_', 'issues']:
            dropped += [k for k in keys_by_tool['sonar'] if term in k]

        # Sonar: Move 'duplicate' metrics to 'Sonar - Duplications' group
        duplications = [k for k in keys_by_tool['sonar'] if 'duplicat' in k]
        keys_by_tool['sonar'] = [k for k in keys_by_tool['sonar'] if k not in duplications]
        keys_by_tool['Sonar - Duplications'] = [
            k.replace('sonar', 'Sonar (Duplications)').replace('duplicated_', '')
            for k in duplications
        ]
        for i, key in enumerate(keys_by_tool['Sonar - Duplications']):
            df_summary[key] = df_summary[duplications[i]]

        # Set index, sort columns, drop path column
        df_summary.drop(columns=dropped, inplace=True, errors='ignore')
        return dropped, df_summary, keys_by_tool

    @classmethod
    def summarize_by_category(
        cls, df: pd.DataFrame, keys_by_tool: dict, output_dir: str | Path
    ) -> tuple:
        """Group various metrics used in the analysis. Save as CSV and LaTeX.

        :param dict keys_by_tool: Columns/metrics grouped by tool.
        :param str | Path output_dir: Path to save the analysis.
        :return tuple: Dropped columns, grouped metrics, and library rankings.
        """
        # fmt: off
        # Group metrics by category
        groups = {
            'Complexity': [
                # 'radon-cc__mean-cc_class', 'radon-cc__mean-cc_function', 'radon-cc__mean-cc_method',
                'radon-cc__total-cc_class', 'radon-cc__total-cc_function', 'radon-cc__total-cc_method',
                'sonar__cognitive_complexity', 'sonar__complexity', 'sonar__file_complexity',
            ],
            'Duplication': keys_by_tool['Sonar - Duplications'],
            'Linting': [
                *[k for k in keys_by_tool['prospector'] if k.endswith('Issues')],
                'pylint__score', 'ruff__Score', 'sonar__errors', 'sonar__violations',
            ],
            'Maintainability and Reliability': [
                'bandit__Total Issues', 'radon-mi__Mean Maintainability Index',  # 'bandit__Issues per Line',
                'sonar__code_smells', 'sonar__development_cost', 'sonar__sqale_debt_ratio', 'sonar__sqale_index',
                'sonar__bugs', 'sonar__reliability_remediation_effort', 'sonar__security_hotspots'
            ],
            'Program Size': [
                # 'radon-mi__Num. Files', # 'radon-raw__Blank Lines', 'radon-raw__LLoC',
                # 'radon-raw__Multi-Line String Lines', 'radon-raw__Single Line Comments',
                'radon-raw__LoC', 'radon-raw__SLoC', 'radon-raw__Comments',
                *[f'radon-hal__Total Halstead {c}' for c in ['Difficulty', 'Time', 'Volume']],
                'sonar__classes', 'sonar__comment_lines_density', 'sonar__files',
                'sonar__functions', 'sonar__ncloc', 'sonar__statements'  # 'sonar__comment_lines', 'sonar__lines',
            ],
            'Test Coverage': [
                'coverage__branch-rate', 'coverage__line-rate', *keys_by_tool['Difference'], 'sonar__line rate',
            ],
        }
        # fmt: on
        # Save category results as CSV and LaTeX
        used_cols: list[str] = []
        median_ranks = {}
        for category, metrics in groups.items():
            try:
                # Filter columns
                df_by_tool = df[[m for m in metrics if m in df.columns]].copy()
                # Sort columns by tool
                used_cols += df_by_tool.columns.to_list()
                # Sort values
                name, ascending = cls.SORT_COLUMNS[category]
                df_by_tool.sort_values(by=name, ascending=ascending, inplace=True)

                # Save as CSV and LaTeX
                cls.save_csv_and_tex(
                    df=df_by_tool,
                    csv_path=Path(output_dir, 'csv', f'summary_{category}.csv'),
                    tex_path=Path(output_dir, 'tables', f'summary_{category}.tex'),
                    output_dir=output_dir,
                    sort_index=df_by_tool.columns.get_loc(name),  # type: ignore
                    category=category,
                )

                if category in ['size', 'Program Size']:
                    continue
                # Record median ranks
                df_by_tool = cls._negate_metric(df_by_tool)
                kwargs = {'ascending': True, 'method': 'average', 'numeric_only': True}
                median_ranks[category] = df_by_tool.median(axis=1).rank(**kwargs)
            except KeyError as e:
                logger.error(f'{e} \ncolumns: {df.columns} \n{category}: {metrics}')
                sys.exit(1)

        # Save overall results as CSV and LaTeX
        logger.debug('Saving overall results...')
        cls.save_csv_and_tex(
            df=df[sorted(df.columns)],
            csv_path=Path(output_dir, 'summary.csv'),
            tex_path=None,
            output_dir=output_dir,
            sort_index=None,
            plot_correlations=False,
        )
        used_cols = sorted(set(used_cols))
        return used_cols, groups, median_ranks

    @classmethod
    def spearman_heatmap(
        cls, df: pd.DataFrame, category: str, output_dir: str | Path, fontsize: int = 12
    ):
        """Compute, save, and plot the Spearman rank correlation matrix.

        :param pd.DataFrame df: The DataFrame being processed.
        :param str category: The category of the metrics.
        :param str | Path output_dir: Path to save the analysis.
        :param int fontsize: Font size for the plot, default is 11.
        """
        # Spearman rank correlation matrix
        df_ranked = df.rank(ascending=False, method='average', numeric_only=True)
        if len(df_ranked.columns) <= 2:
            return

        # Compute and save as CSV
        Path(output_dir, 'correlations').mkdir(parents=True, exist_ok=True)
        matrix = df_ranked.corr(method='spearman')
        csv_path = Path(output_dir, 'correlations', f'spearman_{category}.csv')
        matrix.index.name = 'SCA Metrics'
        matrix.to_csv(csv_path)

        # # Allow importing CSV table into LaTeX
        # with open(csv_path.with_suffix('.tex'), 'w', encoding='utf-8') as f:
        #     newline = '\n\t\\hline\n'
        #     # Start table
        #     column_format = '|r|' + 'c|' * len(df_ranked.columns)
        #     f.write(
        #         '\\begin{table*}[!htbp]\n'
        #         '\\small % \\footnotesize % \\scriptsize\n'
        #         '\\centering\\setlength\\extrarowheight{3pt}\n'
        #         '\\caption{Spearman rank correlation matrix (' + category + ')}\n'
        #         '\\label{tab:spearman_matrix_' + category + '}\n'
        #         '\\begin{tabular}{' + column_format + '}' + newline
        #     )
        #     # First row contains column names
        #     newline = ' \\\\ ' + newline
        #     f.write(f'\t{matrix.index.name} & ' + ' & '.join(matrix.columns) + newline)
        #     # Read contents from CSV file
        #     f.write('\t\\csvreader[head to column names]{' + str(csv_path) + '}{}')
        #     f.write('{\t\\\\ \\hline ' + ' & '.join([f'\\{col}' for col in matrix.columns]) + '}')
        #     # End table
        #     f.write('\\hline \n \\end{tabular} \\end{table*}')
        matrix.style.format(precision=2, thousands='\\,').to_latex(
            str(csv_path).replace('.csv', '.tex'),
            column_format='|r|' + 'c|' * len(df_ranked.columns),
            caption=f'Spearman rank correlation matrix for {category} metrics',
            hrules=True,
            label=f'tab:spearman_{category}',
            position_float='raggedleft',
        )

        # Plot heatmap
        size = 2 + int(len(df_ranked.columns) * 0.75)  # size based on number of columns
        show_tool = 'maintainability' not in category.lower()
        cls.plot_libraries(
            df_ranked, 'spearman', category, output_dir, (size, size), fontsize, show_tool
        )

    @classmethod
    def save_csv_and_tex(
        cls,
        df: pd.DataFrame,
        csv_path: Path,
        tex_path: Path | None,
        output_dir: str | Path,
        sort_index: int | None,
        category: str | None = None,
        plot_correlations: bool = True,
    ) -> pd.DataFrame:
        """Save the DataFrame as a CSV and LaTeX file.

        :param pd.DataFrame df: The DataFrame being processed.
        :param Path csv_path: The path to save the CSV file.
        :param Path | None tex_path: The path to save the LaTeX file.
        :param str | Path output_dir: Path to save the analysis.
        :param int | None sort_index: The index of the column used to sort values.
        :param str | None category: The category of the metrics.
        :param bool plot_correlations: Whether to plot correlations, default is True.
        :return pd.DataFrame: The processed DataFrame.
        """
        # Format column names and numeric values
        df.columns = [cls.format_col_name(c) for c in df.columns.tolist()]
        df = cls.format_numeric_columns(df)

        # Spearman rank correlation matrix
        if plot_correlations:
            cls.spearman_heatmap(df, category if category else 'ALL', output_dir, fontsize=11)

        # Save as CSV
        df.to_csv(csv_path.with_suffix('.csv'), index=True)
        df.T.to_csv(Path(csv_path.parent, 'T_' + csv_path.name), index=True)

        # Multi-index columns
        split_keys = [c.split(': ') for c in df.columns]
        df.columns = pd.MultiIndex.from_tuples(split_keys, names=['Tool', 'Metric'])  # type: ignore
        # Pad first column (library names) to align columns
        df.index = df.index.map(lambda library: library.ljust(cls.LONGEST_NAME))

        # Save as TEX
        if tex_path:
            # Specify LaTeX table caption and label
            if category:
                caption = category.replace('_', ' ').title() + ' metrics for AutoML libraries'
            else:
                caption = tex_path.stem
            # column_format = None  # default format
            column_format = cls.create_column_format(df, sort_index)
            label = f'tab:r:{tex_path.stem}'
            # Save as LaTeX
            df.style.format(precision=2, thousands='\\,', na_rep='-').to_latex(
                tex_path,
                column_format=column_format,
                caption=caption,
                label=label,
                hrules=True,
                multicol_align='r',
                position='!htbp' if len(df.columns) <= 8 else '',
            )
            # Format the LaTeX table for readability and consistency
            if category:
                cls.format_tex_table(df, tex_path, sort_index)
            # Record metric names with bold formatting
            with open(Path(output_dir, 'tex', tex_path.stem + '.tex'), 'w', encoding='utf-8') as f:
                col_names = [col[1].strip() for col in df.columns if col[1] not in '(\%)']
                f.write(', '.join(['\\textit{' + name + '}' for name in col_names]))
        return df

    @classmethod
    def format_numeric_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Format the numeric columns of the DataFrame.

        :param pd.DataFrame df: The DataFrame being processed.
        :return pd.DataFrame: The formatted DataFrame.
        """
        # Round floats
        df = df.round(2)
        # Convert some columns to integer
        for col in df.columns:
            if any(x in col for x in cls.INTEGER_PATTERNS):
                if 'density' in col.lower() or 'per' in col.lower():  # exceptions
                    continue
                if 'num. issues' in col.lower() or 'conditions' in col.lower():
                    df[col].fillna(0, inplace=True)
                df[col] = df[col].astype(int, errors='raise')
        return df

    @classmethod
    def format_col_name(cls, column):
        """Format the column name for readability and consistency.

        :param str column: The column name to format.
        :raises ValueError: If the column name has more than two parts.
        :return str: The formatted column name.
        """
        # Split column name into parts
        parts = column.replace('_', ' ').split('  ')
        if len(parts) > 2:
            raise ValueError(f'Invalid column name "{column}" ({len(parts)} parts)')

        # Format tool name
        tool = (
            parts[0]
            .title()
            .replace('-Raw', '')
            .replace('-Hal', '')
            .replace('-Mi', '')
            .replace('-Cc', ' (CC)')
        )
        # Format metric name
        metric = parts[-1].title() if parts[-1] == parts[-1].lower() else parts[-1]
        metric = (
            metric.replace(' Halstead', '')
            .replace('Sqale', 'SQALE')
            .replace('Ncloc', 'NCLoC')
            .replace('Maintainability Index', 'MI')
            .replace('Reliability Remediation', 'Remediation')
            .replace('-Cc ', ' CC (')
        )
        metric += ')' if 'CC (' in metric else ''
        # Special cases
        if tool == 'Coverage':
            metric = metric.replace('-', ' ')
        if metric.lower() in ['coverage', 'line-rate', 'line rate']:
            # metric = 'Coverage'  # (\\%)
            metric = 'Line Rate'
        if tool == metric and tool.split()[-2:] == ['Num.', 'Issues']:
            tool = tool.split()[0]
            metric = 'Num. Issues'

        return f'{tool}: {metric}'

    @classmethod
    def create_column_format(cls, df: pd.DataFrame, sort_index: int | None) -> str:
        """Create the column format for the LaTeX table.

        :param pd.DataFrame df: The DataFrame being processed.
        :param int | None sort_index: The index of the column used to sort values.
        :return str: The column format for the LaTeX table.
        """
        column_format = 'r|'
        previous_tool = None
        landscape = len(df.columns) > 9
        for tool, metric in df.columns:
            # Add vertical border between tools
            if previous_tool and (landscape or tool != previous_tool):
                column_format += '|'
            previous_tool = tool

            bold = ''
            extra_width = 0.18 if landscape else 0.2
            if tool == 'Difference':
                extra_width += 0.1
            # Sorted columns are bold, wider, and have an arrow
            if sort_index == df.columns.get_loc((tool, metric)):  # type: ignore
                bold = '\\bfseries'
                # extra_width += 0.2

            # Calculate column width based on longest word in metric name or longest value
            width = cls._column_width(df[tool][metric], metric, extra_width)

            # Set width, bold, and align to the right
            column_format += '\n\t\t>{' + bold + '\\raggedleft\\arraybackslash}' + width

        if landscape:
            column_format = '|' + column_format + '|'
        return column_format + '\n\t'

    @classmethod
    def _column_width(cls, series: pd.Series, name: str, extra_width: float = 0.0) -> str:
        """Calculate the column width for a LaTeX table.

        Width is calculated as the maximum of the longest word in the metric name and the maximum series value.

        :param pd.Series series: The column/series being processed.
        :param str name: The name of the series.
        :param float extra_width: Extra width to add to the column, default is 0.0.
        :return str: The column width for the LaTeX table.
        """
        longest = max([len(word) for word in name.split()])
        longest = max(4, longest, len(str(series.max())))
        return 'm{' + str(round(extra_width + (0.135 * longest), 3)) + 'cm}'

    @classmethod
    def format_tex_table(
        cls,
        df: pd.DataFrame,
        tex_path: Path,
        sort_index: int | None,
        text_size: str | None = None,
    ):
        """Format the LaTeX table for readability and consistency.

        :param pd.DataFrame df: The DataFrame being processed.
        :param Path tex_path: The path to the LaTeX table file.
        :param int | None sort_index: The index of the column used to sort values.
        :param str text_size: The font size for the LaTeX table, default is '\\small'.
        """
        # Additional TEX formatting
        with open(tex_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Drop index heading row
        lines = [line for line in lines if not line.startswith('name &  & ')]

        extra_row_height = '3'
        footer_align = 'c'
        if text_size is None:
            text_size = '\\small % \\footnotesize % \\scriptsize\n'

        # Very wide tables are rotated 90 degrees
        if len(df.columns) > 9:
            lines[0] = lines[0].replace('\\begin{table}', '\\begin{sidewaystable*}')
            lines[-1] = lines[-1].replace('\\end{table}', '\\end{sidewaystable*}')
            extra_row_height = '10'
            text_size = '\\footnotesize % \\scriptsize\n'
            footer_align = '|c|'
        # Use table* for wide tables to occupy full width of the page
        elif len(df.columns) > 5:
            lines[0] = lines[0].replace('\\begin{table}', '\\begin{table*}')
            lines[-1] = lines[-1].replace('\\end{table}', '\\vspace{-3mm}\\end{table*}')

        # Format table row by row
        lines, footnotes, end = cls.format_rows(df, lines, sort_index)
        # Add footnotes row if needed
        if len(footnotes) > 0:
            row = [
                '\t' + BORDER,
                '\\multicolumn{' + str(df.shape[1] + 1) + '}{' + footer_align + '}{',
                '\t' + ', '.join(footnotes),
                '} \\\\',
            ]
            lines.insert(end - 1, '\n\t'.join(row))
        # Specify font size, centering, and extra row height
        lines.insert(1, '\\setlength\\extrarowheight{' + extra_row_height + 'pt}\n')
        lines.insert(1, '\\centering\n')
        lines.insert(1, text_size + '\n')
        # Save formatted lines
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        logger.info(f'Saved table to {tex_path}')

    @classmethod
    def format_rows(cls, df: pd.DataFrame, lines: list[str], sort_index: int | None) -> tuple:
        """Format the rows of the LaTeX table for readability and consistency.

        :param pd.DataFrame df: The DataFrame being processed.
        :param list[str] lines: The lines of the LaTeX table.
        :param int | None sort_index: The index of the column used to sort values.
        :return tuple: Formatted lines, footnotes, and the index of the end of tabular
        """
        footnotes: list[str] = []
        index_end = -2  # \end{tabular}
        indent = ''
        col_counter = 0  # Index of column name to be added as a comment
        for i, line in enumerate(lines):
            # Indent table contents for readability
            indents = {'\\toprule': '\t', '\\end{tabular}': ''}
            indent = indents.get(line.strip(), indent)

            # Use hline instead of rules to avoid vertical spacing
            for rule in ['\\toprule', '\\midrule', '\\bottomrule']:
                lines[i] = lines[i].replace(rule, BORDER)

            # Format "Tool" and "Metric" rows in table header
            row_title = line.split('&')[0].strip()
            if row_title in ['Tool', 'Metric']:
                lines[i], footnotes = cls.format_header_row(
                    df, lines[i], sort_index, indent, footnotes
                )

            # Try to align cells for readability
            lines[i] = indent + lines[i].replace(' & ', '\t&\t')

            # Add column names as comments for readability
            if 'cm}' in line:
                lines[i] = lines[i].rstrip() + f' % {df.columns[col_counter]}\n'
                col_counter += 1

            # Record the indices of the 'end' of the tabular environment
            elif '\\end{tabular}' in line:
                index_end = i
        return lines, footnotes, index_end

    @classmethod
    def format_header_row(
        cls, df: pd.DataFrame, row: str, sort_index: int | None, indent: str, footnotes: list[str]
    ) -> tuple:
        """Format the header rows of the LaTeX table.

        :param pd.DataFrame df: The DataFrame being processed.
        :param str row: The row to format.
        :param int | None sort_index: The index of the column used to sort values.
        :param str indent: The current indentation level.
        :param list[str] footnotes: The footnotes to add to the table.
        :return tuple: The formatted row and the updated metrics per tool.
        """
        # Ignore body rows
        row_title = row.split('&')[0].strip()
        if row_title not in ['Tool', 'Metric']:
            return row, footnotes

        # Format the header row title (first cell on left)
        styles = {'Tool': BOLD, 'Metric': ITALIC}
        outer = '|' if len(df.columns) > 9 else ''
        row = row.replace(
            f'{row_title} & ',  # right-align, add vertical border and bold/italicize
            '\\multicolumn{1}{' + outer + 'r|}{' + styles[row_title] + '{' + row_title + '}} & ',
        )

        # Format remaining header row cells
        # Center-align tool names, add vertical borders
        row = row.replace('{r}', '{c|}')

        # Column names in row: center-align, bold/italicize, word wrap
        cols = row.split('&')[1:]
        seen = set()
        for i, col in enumerate(cols):
            col = col.replace('\\\\', '').strip()
            arrow = ''
            if row_title == 'Tool':
                if 'multicolumn' in col:
                    # Get tool name
                    parts = [s.replace('}', '') for s in col.split('{')[1:]]
                    col = parts[2].strip()
                else:
                    # Center-align, add vertical border
                    align = 'c' if 'Sonar' in col or i > len(cols) - 1 else 'c|'
                    row = row.replace(col, ' \\multicolumn{1}{' + align + '}{' + col + '} ')
            else:
                # # Add arrow to the column name if used for sorting
                # if sort_index and i == sort_index: arrow = ' $\\downarrow$'
                # Bold/italicize
                if col not in seen:
                    row = row.replace(f' {col} ', ' ' + ITALIC + '{' + col + arrow + '} ')
                    seen.add(col)
            # Fix errors
            row = row.replace('File \\textit{Complexity}', '\\textit{File Complexity}')

        # Add border
        if row_title == 'Tool':
            row += indent + BORDER + '\n'

        # Check if footnotes are needed
        for key, footnote in cls.FOOTNOTES.items():
            if key.lower() in row.lower():
                footnotes.append(footnote)
        return row, footnotes

    @classmethod
    def word_wrap(cls, row: str, col: str, column_format: str | None = None, max_length: int = 12):
        """Wrap long words in a LaTeX table cell.

        :param str row: The row to format.
        :param str col: The cell content.
        :param str | None column_format: The column format for the LaTeX table.
        :param int max_length: The maximum length of a word before wrapping.
        :return tuple: The formatted row and cell content.
        """
        if len(col) > max_length and ' ' in col:
            # Wrap long words
            wrapped_text = textwrap.wrap(col, width=max_length, break_long_words=False)
            # Depending on the column format, use newline or vtop/hbox/strut
            if column_format and 'cm}' in column_format:
                # row = row.replace(col, ' \\newline '.join(wrapped_text))
                pass
            else:
                # Required markup for multi-line text
                row = row.replace(col, f'\\vtop{{\\hbox{{\\strut {col}}}}}')
                # Word wrap column name using latex instead of newline
                row = row.replace(col, '}\\hbox{\\strut '.join(wrapped_text))
        return row, col

    @classmethod
    def create_plots(cls, df_summary: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
        """Plot the results of the SCA analysis.

        :param pd.DataFrame df_summary: The results as a pandas DataFrame.
        :param str | Path output_dir: Path to save the analysis.#
        :return pd.DataFrame: Median ranks of the libraries.
        """
        corr_dir = Path(output_dir, 'correlations')
        corr_dir.mkdir(parents=True, exist_ok=True)
        # Bar plot of LoC
        cols = ['coverage__lines-valid', 'sonar__statements']
        df_loc = df_summary[cols].sort_index(ascending=False)
        df_loc.columns = ['Coverage.py "Lines Valid"', 'Sonar "Statements"']
        cls.plot_libraries(df_loc, 'barh', 'Lines of Code', output_dir, (6, 5))

        # Bar plot of 'coverage__line-rate' and 'sonar__line rate'
        cols = ['coverage__line-rate', 'sonar__line rate']
        df_coverage = df_summary[cols].sort_index(ascending=False)
        df_coverage.columns = ['Coverage.py', 'SonarQube']
        cls.plot_libraries(df_coverage, 'barh', 'Reported Line Coverage', output_dir, (6, 5))

        # Invert some scores so that higher values indicate 'worse' results
        df_ranks = df_summary[cls.RANK_METRICS]
        df_ranks = cls._negate_metric(df_ranks)

        # Calculate library ranks across all metrics
        kwargs = {'ascending': True, 'method': 'average', 'numeric_only': True}
        df_ranks = df_ranks.rank(**kwargs)
        df_ranks['Median Rank'] = df_ranks.median(axis=1)

        # Calculate library ranks for Sonar metrics only
        sonar_metrics = [col for col in df_ranks.columns if 'sonar' in col]
        df_ranks['M. Rank (Sonar-Only)'] = df_ranks[sonar_metrics].rank(**kwargs).median(axis=1)

        # Calculate library ranks for non-Sonar metrics
        non_sonar_metrics = [col for col in df_ranks.columns if col not in sonar_metrics]
        df_ranks['M. Rank (Non-Sonar)'] = df_ranks[non_sonar_metrics].rank(**kwargs).median(axis=1)

        # Sort by mean rank and save to CSV
        df_ranks.sort_values(by='Median Rank', inplace=True)
        df_ranks.to_csv(Path(output_dir, 'csv', 'ranks.csv'))
        df_ranks.to_csv(Path(corr_dir, 'ranks.csv'))

        # Box plot of ranks
        df_melted = (
            df_ranks[cls.RANK_METRICS]
            .reset_index()
            .rename(columns={'name': 'Library'})
            .melt(id_vars=['Library'], var_name='Median Rank')
        )
        title = 'AutoML library ranks for all SCA metrics'
        cls.plot_libraries(df_melted, 'box', title, output_dir, (10, 5))

        # Plot spearman rank correlation matrix
        df_ranks.corr(method='spearman').to_csv(Path(corr_dir, 'correlation_rank.csv'))
        return df_ranks

    @classmethod
    def _negate_metric(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Negate the values of certain metrics to indicate 'worse' results.

        :param pd.DataFrame df: The DataFrame being processed.
        :return pd.DataFrame: The processed DataFrame.
        """
        for col in [
            'coverage__branch-rate',
            'coverage__line-rate',
            'pylint__score',
            'radon-mi__Mean Maintainability Index',
            'sonar__coverage',
            'sonar__line rate',
        ]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: -x if x > 0 else x)
        return df

    @classmethod
    def plot_libraries(
        cls,
        data: pd.DataFrame,
        kind: str,
        metric: str,
        output_dir: str | Path,
        figsize: tuple[int, int] = (10, 8),
        fontsize: int = 12,
        show_tool: bool = True,
        **kwargs,
    ):
        """Plot the results of the SCA analysis.

        :param pd.DataFrame | pd.Series data: The data to plot.
        :param str kind: The kind of plot to create.
        :param str metric: Metric used to label the plot.
        :param str | Path output_dir: Path to save the analysis.
        :param tuple[int, int] figsize: The size of the plot.
        :param int fontsize: The font size of the plot.
        :param bool show_tool: Whether to show the tool name in the plot.
        """
        # Set colorblind-friendly style
        plt.style.use('tableau-colorblind10')
        # Set figure size
        _, ax = plt.subplots(figsize=figsize)

        # Create plot
        if kind == 'box':  # Seaborn box plot
            file_name = cls._plot_box(data, metric, fontsize, ax)
        elif kind == 'barh':
            file_name = cls._plot_bar(data, metric, fontsize, ax, figsize)
        elif kind in ['spearman']:
            file_name = cls._plot_heatmap(data, metric, fontsize, ax, show_tool, kind, **kwargs)
        elif kind == 'heatmap':
            file_name = cls._plot_heatmap(data, metric, fontsize, ax, show_tool, None, **kwargs)
        else:
            raise ValueError(f'Invalid plot kind: {kind}')

        # Save plot to file
        file_name = file_name.replace('(\\%)', '').replace('(%)', '')
        path = Path(output_dir, 'plots', f'{file_name}.svg')
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        # plt.show()
        # Close plot
        plt.cla()
        plt.clf()
        plt.close('all')

    @classmethod
    def _plot_box(cls, data: pd.DataFrame, metric: str, fontsize: int, ax: plt.Axes) -> str:
        """Plot a box plot of the data.

        :param pd.DataFrame data: The data to plot.
        :param str metric: Metric used to label the plot.
        :param int fontsize: The font size of the plot.
        :param plt.Axes ax: The axes of the plot.
        :return str: File name of the plot.
        """
        sns.boxplot(
            data=data,
            y='value',
            x='Library',
            # orient='h',
            fill=False,
            hue='Library',
            palette=sns.dark_palette('seagreen', n_colors=data.shape[0]),
            ax=ax,
        )
        y_ticks = [int(label.get_text()) for label in ax.get_yticklabels()]
        y_ticks = list(range(min(y_ticks) + 1, max(y_ticks) - 1))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(t) for t in y_ticks], fontsize=fontsize - 2)  # type: ignore
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=fontsize - 2)
        plt.xlabel(metric, fontsize=fontsize)
        plt.ylabel('SCA Metric Rank', fontsize=fontsize)
        plt.tight_layout()
        return f'boxplot_{metric}'

    @classmethod
    def _plot_bar(
        cls, data: pd.DataFrame, metric: str, fontsize: int, ax: plt.Axes, figsize: tuple[int, int]
    ) -> str:
        """Plot a bar plot of the data.

        :param pd.DataFrame data: The data to plot.
        :param str metric: Metric used to label the plot.
        :param int fontsize: The font size of the plot.
        :param plt.Axes ax: The axes of the plot.
        :param tuple[int, int] figsize: The size of the plot.
        :return str: File name of the plot.
        """
        data.plot(
            ax=ax,
            kind='barh',  # type: ignore
            figsize=figsize,
            # title=f'AutoML library results for {metric}',
            fontsize=fontsize,
            edgecolor='black',
            linewidth=1,
            colormap='cividis',
        )
        plt.xlabel(metric, fontsize=fontsize)
        # plt.ylabel('Library', fontsize=fontsize)
        ax.get_yaxis().get_label().set_visible(False)
        plt.tight_layout()
        return f'barh_{metric}'

    @classmethod
    def _plot_heatmap(
        cls,
        data: pd.DataFrame,
        metric: str,
        fontsize: int,
        ax: plt.Axes,
        show_tool: bool,
        correlation: str | None = 'spearman',
        compact: bool = True,
        show_p_values: bool = True,
        p_value_threshold: float = 0.05,
        **kwargs,
    ) -> str:
        """Plot a heatmap of the Spearman rank correlation matrix.

        :param pd.DataFrame data: The data to plot.
        :param str metric: Metric used to label the plot.
        :param int fontsize: The font size of the plot.
        :param plt.Axes ax: The axes of the plot.
        :param bool show_tool: Whether to show the tool name in the plot.
        :param str | None correlation: The type of correlation to compute, default is 'spearman'.
        :param bool compact: Whether to use a compact format for the plot.
        :param bool show_p_values: Whether to show p-values in the plot.
        :return str: File name of the plot.
        """
        large_matrix = len(data.columns) > 20  # May require different formatting

        # Spearman rank correlation matrix and p-values
        p_values, mask = None, None
        fmt = '.0f'
        square = True
        if correlation:
            correlations = data.corr(method=correlation)  # type: ignore
            p_values = data.corr(method=lambda x, y: spearmanr(x, y)[1])  # type: ignore
            # mask upper triangle
            mask = np.triu(np.ones_like(correlations, dtype=bool), k=1 if large_matrix else 0)
            data = correlations
            fmt = '.2f'
            square = False

            if large_matrix:
                # Also mask cells where p-values greater than p_value_threshold
                mask = np.where(p_values > p_value_threshold, True, mask)
                # mask_alt = np.where(p_values <= p_value_threshold, True, mask)

        # Plot heatmap
        sns.set_theme(context='paper', style='white', font_scale=1, font='serif')  # sans-serif
        annot_kwargs = {
            'size': fontsize,
            'color': 'black',
            'weight': 'bold',
            'va': 'center',
            'ha': 'center',
        }
        heatmap_kwargs = {
            'data': data,
            'annot': True,
            'ax': ax,
            'cbar': False,
            'center': 0,
            'fmt': fmt,
            'linewidths': 0.5 if large_matrix else 0.1,
            'linecolor': 'white' if compact else 'grey',
            'square': square,
        }
        # Add alternative heatmap for insignificant p-values
        annot_kwargs['weight'] = 'bold'
        ax = sns.heatmap(
            annot_kws=annot_kwargs,
            cmap=sns.diverging_palette(5, 250, l=75, as_cmap=True),
            mask=mask,
            yticklabels=data.index,  # type: ignore
            xticklabels=data.columns,  # type: ignore
            **heatmap_kwargs,
        )
        # if large_matrix:
        #     annot_kwargs['weight'] = 'normal'
        #     annot_kwargs['ax'] = ax
        #     sns.heatmap(
        #         annot_kws=annot_kwargs,
        #         mask=mask_alt,
        #         yticklabels=False,
        #         xticklabels=False,
        #         **heatmap_kwargs,
        #     )

        if large_matrix:
            # Add grid lines to the heatmap
            lines = [i * 5 for i in range(0, (data.shape[0] // 5) + 1)]
            ax.hlines(lines, *ax.get_ylim(), color='black')
            ax.vlines(lines, *ax.get_xlim(), color='black')

            # Replace x and y labels with numbers as matrix is too big
            labels = [
                f'{label.get_text()} - ({i})' for i, label in enumerate(ax.get_xticklabels(), 1)
            ]
            ax.set_xticklabels(
                labels, ha='right', fontsize=fontsize + 2, rotation=90, fontdict={'weight': 'bold'}
            )
            labels = [f'({i})' for i in range(1, len(data.index) + 1)]
            # ax.tick_params(axis='y', labeltop=True, labelright=True, rotation=0)
            ax.set_yticklabels(labels, fontsize=fontsize + 1, fontdict={'weight': 'bold'})
        else:
            # Set axis labels
            if isinstance(correlation, str):
                # Format axis labels and remove diagonal
                x_labels = cls._shorten_metric_names(ax.get_xticklabels(), show_tool)
                y_labels = cls._shorten_metric_names(ax.get_yticklabels(), show_tool)
                if compact:
                    x_labels[-1], y_labels[0] = '', ''
            else:
                x_labels = [label.get_text() for label in ax.get_xticklabels()]
                y_labels = [label.get_text() for label in ax.get_yticklabels()]

            kwargs = {'fontsize': fontsize, 'ha': 'right', 'fontdict': {'weight': 'bold'}}
            ax.set_xticklabels(x_labels, rotation=45, **kwargs)  # type: ignore
            ax.set_yticklabels(y_labels, rotation=0, **kwargs)  # type: ignore

        # Add p-value annotations
        if isinstance(correlation, str) and show_p_values:
            cls._p_value_annotations(p_values, data, ax, fontsize, compact)  # type: ignore

        # plt.title(f'Spearman rank correlation matrix for {metric}')
        correlation = correlation + '_' if isinstance(correlation, str) else ''
        return f'{correlation}heatmap_{metric}'

    @classmethod
    def _shorten_metric_names(cls, names: list, show_tool: bool) -> list[str]:
        """Shorten metric names.

        :param list names: The list of metric names to shorten.
        :param bool show_tool: Whether to show the tool name in the plot.
        :return list[str]: The shortened metric names.
        """
        aliases = {
            'Coverage': 'Cov.',
            'Radon': '',
            'Radon (CC)': '',
            'Radon (Halstead)': 'Halstead',
            'Sonar': 'Sonar',
            'Sonar (Duplications)': 'Duplicated',
        }

        for i, name in enumerate(names):
            name = name.get_text().split(': ')
            if len(name) == 2:
                tool, metric = name
                tool = aliases.get(tool, tool) if show_tool else ''
            else:
                tool, metric = '', name[0]
            # Format metric name
            metric = metric.replace('Complexity', 'Com.').replace('Cognitive', 'Cog.')
            metric = metric.replace('Num. Issues', 'Issues')
            names[i] = f'{tool} {metric}'
        return names

    @classmethod
    def _p_value_annotations(
        cls,
        p_values: pd.DataFrame,
        correlations: pd.DataFrame,
        ax: plt.Axes,
        fontsize: int,
        compact: bool,
        p_value_threshold: float = 0.05,
    ):
        """Assign p-value annotations.

        :param pd.DataFrame p_values: The p-values DataFrame.
        :param pd.DataFrame correlations: The correlations DataFrame.
        :param plt.Axes ax: The axes of the plot.
        :param int fontsize: The font size for the annotations.
        :param bool compact: Whether to use a compact format for the annotations.
        :param float p_value_threshold: The threshold for p-values to be considered significant.
        """
        large_matrix = len(p_values.columns) > 20
        for row in range(p_values.shape[0]):
            for col in range(p_values.shape[1]):
                # Skip diagonal and upper triangle, i.e. compact -> p-values in lower triangle
                if compact and row >= col:
                    continue
                # Skip diagonal and lower triangle, i.e. not compact -> p-values in upper triangle
                elif not compact and row <= col:
                    continue

                # Skip NaN p-values and perfect correlations
                p_value = float(p_values.iloc[row, col])  # type: ignore
                correlation = correlations.iloc[row, col]  # type: ignore
                if np.isnan(p_value) or correlation == 1:
                    continue

                # Format p-value string
                significance = 'bold' if p_value < p_value_threshold else 'normal'
                if large_matrix:
                    if p_value > p_value_threshold:  # Skip annotation in large matrices
                        continue
                    text = f'{p_value:.3f}'.replace('0.', '.')
                    fontdict = {'fontsize': fontsize, 'weight': significance, 'color': 'black'}
                    x, y = row + 0.5, col + 0.5
                else:
                    text = f'p={p_value:.4f}'
                    fontdict = {'fontsize': fontsize - 1, 'weight': significance, 'color': 'black'}
                    x, y = row + 0.5, col + 0.8

                # Add p-value annotation
                ax.text(x=x, y=y, s=text, ha='center', va='center', fontdict=fontdict)

    @classmethod
    def save_notes_as_tex(
        cls, df_summary: pd.DataFrame, df_ranks_summary: pd.DataFrame, output_dir: str | Path
    ):
        """Save notes as LaTeX files.

        :param pd.DataFrame df_summary: Summary DataFrame with metrics.
        :param pd.DataFrame df_ranks_summary: Ranks summary DataFrame with median ranks.
        :param str | Path output_dir: Directory to save the tex files.
        """
        # Record sortable metrics grouped by tool
        cls._record_sortable_metrics(df_summary, output_dir)

        # Record libraries in order of median ranks
        cls._record_libraries_median_ranks(df_ranks_summary, output_dir)

        # The correlation between "Median Rank (Sonar-Only)" and "Median Rank (Non-Sonar)" is:
        sonar = df_ranks_summary['M. Rank (Sonar-Only)']
        non_sonar = df_ranks_summary['M. Rank (Non-Sonar)']
        correlation, p_value = spearmanr(sonar, non_sonar)
        with open(Path(output_dir, 'tex', 'corr_sonar_non_sonar.tex'), 'w', encoding='utf-8') as f:
            f.write(
                'The Spearman rank correlation between Sonar and non-Sonar rankings is '
                f'{correlation:.4f} (p-value$\\approx${p_value:.6f}).'
            )

    @classmethod
    def _record_sortable_metrics(cls, df_summary: pd.DataFrame, output_dir: str | Path):
        """Record sortable metrics grouped by tool.

        :param pd.DataFrame df_summary: Summary DataFrame with metrics.
        :param str | Path output_dir: Directory to save the tex files.
        """
        sortable_metrics = [
            cls.format_col_name(m) for m in cls.RANK_METRICS if m in df_summary.columns
        ]
        previous_tool = None
        text = ''
        for name in sortable_metrics:
            tool, metric = name.split(': ')
            tool = tool.replace(' (CC)', '')
            if tool != previous_tool:
                if previous_tool:
                    text += '), '
                text += '\\textbf{' + tool + '} ('
                previous_tool = tool
            else:
                text += ', '
            text += '\\textit{' + metric + '}'
        text += ').'

        with open(Path(output_dir, 'tex', 'sortable_metrics.tex'), 'w', encoding='utf-8') as f:
            f.write(text)

    @classmethod
    def _record_libraries_median_ranks(cls, df_ranks_summary: pd.DataFrame, output_dir: str | Path):
        """Record libraries in order of median ranks.

        :param pd.DataFrame df_ranks_summary: Ranks summary DataFrame with median ranks.
        :param str | Path output_dir: Directory to save the tex files.
        """
        text = ''
        text_best = ''
        text_worst = ''
        for i, library in enumerate(df_ranks_summary.index):
            # Add commas
            if len(text) > 0 and i < len(df_ranks_summary.index) - 1:
                text += ', '
                if i < 5:
                    text_best += ', '
                if i > len(df_ranks_summary.index) - 5:
                    text_worst += ', '
            # "and"
            if i == len(df_ranks_summary.index) - 1:
                text += ', and '
                text_worst += ', and '
            elif i == 4:
                text_best += ' and '

            text += '\\code{' + library + '}'
            if i < 5:
                text_best += '\\code{' + library + '}'
            if i > len(df_ranks_summary.index) - 6:
                text_worst += '\\code{' + library + '}'
        text_best += ' are the best-ranked libraries.'
        text_worst += ' are the worst-ranked libraries.'
        text += '.'

        for text_, path_ in [
            (text, 'libraries_sorted_by_median.tex'),
            (text_best, 'best_libraries_sorted_by_median.tex'),
            (text_worst, 'worst_libraries_sorted_by_median.tex'),
        ]:
            with open(Path(output_dir, 'tex', path_), 'w', encoding='utf-8') as f:
                f.write(text_)
