"""Functionality for reporting the results of source code analysis."""

import json
import textwrap
from pathlib import Path

import pandas as pd

from ml import LIBRARIES
from ml.logs import logger


class Reporting:
    """SCA reporting functionality."""

    # LaTeX formatting
    BOLD = '\\textbf'
    ITALIC = '\\textit'
    BORDER = '\\hline'

    FOOTNOTES = {
        'CC': ITALIC + '{ CC: Cyclomatic Complexity}',
        'LLoC': ITALIC + '{ LLoC: Logical Lines of Code}',
        ' LoC': ITALIC + '{ LoC: Lines of Code}',
        'NCLoC': ITALIC + '{ NCLoC: Non-Comment Lines of Code}',
        'SLoC': ITALIC + '{ SLoC: Source Lines of Code}',
        'SQALE': ITALIC + '{ SQALE: Software Quality Assessment based on Lifecycle Expectations}',
    }
    LONGEST_NAME = max(len(name) for name in LIBRARIES.values())
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

        # Ensure correct library names
        df['name'].replace(LIBRARIES, inplace=True)

        # Save all results to a CSV file
        results_file = Path(output_dir, 'results.csv')
        df.to_csv(results_file, index=False)
        logger.info(f'Saved results to {results_file}')

        # Filter, combine and/or format columns
        df_summary, keys_by_tool = cls.filter_format_columns(df, ignored_keys_by_tool)

        # Summarize results and export to CSV, Markdown, and LaTeX
        dropped_cols, groups = cls.summarize_tools(df_summary, keys_by_tool, output_dir)
        used_cols = [col for group in groups.values() for col in group]
        unused_cols = [col for col in df_summary.columns if col not in used_cols]

        # Save metadata to a JSON file
        metadata = {
            'metrics_dropped': dropped_cols,
            'metrics_ignored': ignored_keys_by_tool,
            'metrics_used': used_cols,
            'metrics_unused': unused_cols,
            'num_repos': len(df),
        }
        metadata_file = Path(output_dir, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f'Saved metadata to {metadata_file}')

        return df

    @classmethod
    def filter_format_columns(cls, df: pd.DataFrame, ignored_keys_by_tool: dict[str, list[str]]):
        """Filter and format the columns of the results dataframe for analysis.

        :param pd.DataFrame df: The results as a pandas DataFrame.
        :param dict[str, list[str]] ignored_keys_by_tool: Ignored keys by tool.
        :return tuple: The filtered and formatted DataFrame and the columns grouped by tool.
        """
        # Group keys by tool
        keys_by_tool: dict[str, list[str]] = {}
        for col in df.columns:
            if '__' in col:
                tool = col.split('__')[0]
                keys_by_tool[tool] = keys_by_tool.get(tool, []) + [col]

        # Bandit: score = sum of all bandit issues
        metric_keys = [key for key in keys_by_tool['bandit'] if key.startswith('bandit__B')]
        df['bandit__Total Issues'] = df[metric_keys].sum(axis=1)
        df['bandit__Issues per Line'] = (  # Bandit score per line
            df['bandit__Total Issues'] / df['bandit__loc']
        )
        df_summary = df.drop(columns=keys_by_tool['bandit'])

        # Coverage: drop redundant coverage metrics
        dropped_keys = [f'coverage__{key}' for key in ignored_keys_by_tool['coverage']]
        df_summary.drop(columns=dropped_keys, inplace=True, errors='ignore')

        # Prospector: only keep score
        dropped_keys = [k for k in keys_by_tool['prospector'] if not k.endswith(' Score')]
        df_summary.drop(columns=dropped_keys, inplace=True)

        # Pylint: only keep score
        dropped_keys = [k for k in keys_by_tool['pylint'] if k != 'pylint__score']
        df_summary.drop(columns=dropped_keys, inplace=True)

        # Radon CC: only keep sum and mean
        dropped_keys = [
            k for k in keys_by_tool['radon-cc'] if 'total-' not in k and 'mean-' not in k
        ]
        df_summary.drop(columns=dropped_keys, inplace=True, errors='ignore')

        # Ruff: only keep score
        df_summary['ruff__Ruff Score'] = df_summary['ruff__ruff Score']
        df_summary.drop(columns=keys_by_tool['ruff'], inplace=True, errors='ignore')

        # Sonar: drop 'new' and 'issues' metrics
        for term in ['_new_', 'issues']:
            dropped_keys = [k for k in keys_by_tool['sonar'] if term in k]
            df_summary.drop(columns=dropped_keys, inplace=True, errors='ignore')

        # Set index, sort columns, drop path column
        df_summary.set_index('name', drop=True, inplace=True)
        df_summary.drop(columns=['path'], inplace=True, errors='ignore')
        df_summary = df_summary[sorted(df_summary.columns)]
        return df_summary, keys_by_tool

    @classmethod
    def summarize_tools(cls, df: pd.DataFrame, keys_by_tool: dict, output_dir: str | Path) -> tuple:
        """Group various metrics used in the analysis. Save as CSV and LaTeX.

        :param dict keys_by_tool: Columns/metrics grouped by tool.
        :param str | Path output_dir: Path to save the analysis.
        :return tuple: Dropped columns and grouped metrics.
        """
        # fmt: off
        # Group metrics by category
        groups = {
            'complexity': [
                'radon-cc__mean-cc_class', 'radon-cc__mean-cc_function', 'radon-cc__mean-cc_method',
                # 'radon-cc__total-cc_class', 'radon-cc__total-cc_function', 'radon-cc__total-cc_method',
                'sonar__cognitive_complexity', 'sonar__complexity', 'sonar__file_complexity',
                'coverage__complexity',
            ],
            'counts': [
                'git__LoC',  # Git
                'radon-raw__LLoC', 'radon-raw__LoC', 'radon-raw__SLoC',  # Radon
                'radon-raw__Multi-Line String Lines', 'radon-raw__Single Line Comments',
                'radon-raw__Comments', 'radon-raw__Blank Lines',
                'sonar__ncloc', 'sonar__ncloc_language_distribution', 'sonar__lines',  # Sonar
                'sonar__comment_lines', 'sonar__comment_lines_density', 'sonar__files',
                'sonar__functions', 'sonar__statements', 'sonar__classes',
            ],
            'coverage': [col for col in df.columns if 'cover' in col],
            'duplication': [col for col in df.columns if 'duplicat' in col],
            'git': ['git__Num. Commits', 'git__Num. Contributors'],
            'halstead_mean': [col for col in df.columns if 'Mean Halstead' in col],
            'halstead_total': [col for col in df.columns if 'Total Halstead' in col],
            'linting': [
                *[k for k in keys_by_tool['prospector'] if k.endswith(' Score')],
                'pylint__score', 'ruff__Ruff Score',
                'sonar__bugs', 'sonar__code_smells', 'sonar__errors', 'sonar__violations',
            ],
            'maintainability': [
                *keys_by_tool['radon-mi'],
                'sonar__effort_to_reach_maintainability_rating_a', 'sonar__development_cost',
                'sonar__sqale_debt_ratio', 'sonar__sqale_index', 'sonar__sqale_rating',
            ],
            'reliability': ['sonar__reliability_rating', 'sonar__reliability_remediation_effort'],
            'security': [
                'bandit__Total Issues', 'bandit__Issues per Line', 'sonar__security_hotspots'
            ],
        }
        # fmt: on
        dropped_cols = []
        filename = Path(output_dir, 'summary.csv')
        for category, metrics in groups.items():
            csv_path = filename.with_name(f'summary_{category}.csv')
            try:
                # Filter and rename columns
                metrics = [m for m in metrics if m in df.columns]
                df_by_tool = df[metrics].copy()
                df_by_tool.columns = [c.replace('prospector__', '') for c in df_by_tool.columns]

                # Drop columns with only one unique value
                counts = df_by_tool.nunique()
                dropped = [col for col in counts.index.to_list() if counts[col] < 2]
                if len(dropped) > 0:
                    logger.warning(f'Dropping columns with only one unique value: {dropped}')
                df_by_tool.drop(columns=dropped, inplace=True)
                dropped_cols += dropped

                # Save as CSV and LaTeX
                cls.save_csv_and_tex(df_by_tool, csv_path, category)
            except KeyError as e:
                logger.error(f'{e} \ncolumns: {df.columns} \n{category}: {metrics}')
                exit(1)

        # Save as CSV and LaTeX
        cls.save_csv_and_tex(df.copy(), filename)

        return dropped_cols, groups

    @classmethod
    def save_csv_and_tex(cls, df: pd.DataFrame, file_path: Path, category: str | None = None):
        """Save the DataFrame as a CSV and LaTeX file.

        :param str | Path file_path: The file path to save the DataFrame as a CSV and LaTeX file.
        """
        df = df.round(2)
        # Save as CSV
        df.columns = [cls.format_col_name(c) for c in df.columns.tolist()]
        df.to_csv(file_path.with_suffix('.csv'), index=True)
        logger.info(f'Saved "{category}" results to {file_path.with_suffix(".csv")}')

        # Convert some columns to integer
        for col in df.columns:
            if any([x in col for x in ['Num.', 'oC', 'spots', 'Duplicated', 'Issues', 'Lines']]):
                if 'Density' in col or 'per' in col:
                    continue
                df[col] = df[col].astype(int)

        # Multi-index columns
        split_keys = [c.split(': ') for c in df.columns]
        df.columns = pd.MultiIndex.from_tuples(split_keys, names=['Tool', 'Metric'])

        # Pad first column (library names) to align columns
        df.index = df.index.map(lambda library: library.ljust(cls.LONGEST_NAME))

        # Save as TEX
        tex_path = file_path.with_suffix('.tex')
        caption = f'{category.title()} metrics for AutoML libraries' if category else tex_path.stem
        df.style.format(precision=2, thousands=',', na_rep='-').to_latex(
            file_path.with_suffix('.tex'),
            caption=caption,
            label=f'tab:{file_path.stem}',
            hrules=True,
            multicol_align='r',
            position='!htbp',
        )
        # Format the LaTeX table for readability and consistency
        if category:
            cls.format_tex_table(df, tex_path)

    @classmethod
    def format_col_name(cls, column):
        """Format the column name for readability and consistency.

        :param str column: The column name to format.
        :raises ValueError: If the column name has more than two parts.
        :return str: The formatted column name.
        """
        parts = column.replace('_', ' ').split('  ')
        if len(parts) > 2:
            raise ValueError(f'Invalid column name "{column}" ({len(parts)} parts)')

        # Format tool name
        tool = parts[0].title()
        tool = tool.replace('-Raw', '').replace('-Hal', ' (Halstead)')
        tool = tool.replace('-Mi', '').replace('-Cc', '')

        # Format metric name
        metric = parts[-1].title() if parts[-1] == parts[-1].lower() else parts[-1]
        metric = metric.replace('-Cc', ' CC by').replace(' Halstead', '')
        metric = metric.replace('Sqale', 'SQALE').replace('Ncloc', 'NCLoC')
        if tool == 'Coverage':
            metric = metric.replace('-', ' ')

        return f'{tool}: {metric}'

    @classmethod
    def format_tex_table(cls, df, tex_path: Path):
        """Format the LaTeX table for readability and consistency.

        :param pd.DataFrame df: The DataFrame being processes.
        :param Path tex_path: The path to the LaTeX table file.
        """
        # Additional TEX formatting
        with open(tex_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Drop index heading row
        lines = [line for line in lines if not line.startswith('name &  & ')]

        # Use table* for wide tables
        lines[0] = lines[0].replace('\\begin{table}', '\\begin{table*}')
        lines[-1] = lines[-1].replace('\\end{table}', '\\end{table*}')

        # Format table row by row
        lines, metrics_per_tool, footnotes, begin, end = cls.format_rows(lines)

        # Add vertical borders to categorize metrics by tool
        lines[begin] = lines[begin].replace('{l', '{|r|')
        offset = len('\\begin{tabular}{|r|')
        for i, count in enumerate(metrics_per_tool):
            offset += count + i
            lines[begin] = lines[begin][:offset] + '|' + lines[begin][offset:]

        # Add footnotes row if needed
        if len(footnotes) > 0:
            row = [
                '\t' + cls.BORDER,
                '\\multicolumn{' + str(df.shape[1] + 1) + '}{|l|}{',
                '\t' + ', '.join(footnotes),
                '} \\\\',
            ]
            lines.insert(end - 1, '\n\t'.join(row))

        # Specify font size, centering, and extra row height
        lines.insert(1, '\\setlength\\extrarowheight{3pt} % extra row height\n')
        lines.insert(1, '\\centering\n')
        lines.insert(1, '\\small % \\footnotesize % \\scriptsize\n')

        # Save formatted lines
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        logger.info(f'Saved results to {tex_path}')

    @classmethod
    def format_rows(cls, lines: list[str]):
        """Format the rows of the LaTeX table for readability and consistency.

        :param list[str] lines: The lines of the LaTeX table.
        :return list[str]: The formatted lines of the LaTeX table.
        """
        footnotes: list[str] = []
        metrics_per_tool: list[int] = []  # i.e. where to add vertical borders
        index_begin = 3  # \begin{tabular}
        index_end = -2  # \end{tabular}
        indent = ''
        for i, line in enumerate(lines):
            # Indent table contents for readability
            indents = {'\\toprule': '\t', '\\end{tabular}': ''}
            indent = indents.get(line.strip(), indent)

            # Use hline instead of rules to avoid vertical spacing
            for rule in ['\\toprule', '\\midrule', '\\bottomrule']:
                lines[i] = lines[i].replace(rule, cls.BORDER)

            # Format "Tool" and "Metric" rows in table header
            row_title = line.split('&')[0].strip()
            if row_title in ['Tool', 'Metric']:
                lines[i], metrics_per_tool, footnotes = cls.format_header_row(
                    lines[i], metrics_per_tool, indent, footnotes
                )

            # Try to align cells for readability
            lines[i] = indent + lines[i].replace(' & ', '\t&\t')

            # Record the indices of the 'begin' and 'end' of the tabular environment
            if '\\begin{tabular}' in line:
                index_begin = i
            elif '\\end{tabular}' in line:
                index_end = i
        return lines, metrics_per_tool, footnotes, index_begin, index_end

    @classmethod
    def format_header_row(
        cls, row: str, metrics_per_tool: list[int], indent: str, footnotes: list[str]
    ) -> tuple:
        """Format the header rows of the LaTeX table.

        :param str row: The row to format.
        :param list[int] metrics_per_tool: The number of metrics per tool.
        :param str indent: The current indentation level.
        :param list[str] footnotes: The footnotes to add to the table.
        :return tuple: The formatted row and the updated metrics per tool.
        """
        # Ignore body rows
        row_title = row.split('&')[0].strip()
        if row_title not in ['Tool', 'Metric']:
            return row, metrics_per_tool, footnotes

        # Format the header row title (first cell on left)
        styles = {'Tool': 'textbf', 'Metric': 'textit'}
        row = row.replace(
            f'{row_title} & ',  # right-align, add vertical border and bold/italicize
            '\\multicolumn{1}{|r|}{\\' + styles[row_title] + '{' + row_title + '}} & ',
        )

        # Format remaining header row cells
        row = row.replace('{r}', '{c|}')  # Center-align, add vertical borders

        # Column names in row: center-align, bold/italicize, word wrap
        for col in row.split('&')[1:]:
            col = col.replace('\\\\', '').strip()
            # Get multicolumn sizes
            if row_title == 'Tool':
                if 'multicolumn' in col:
                    parts = [s.replace('}', '') for s in col.split('{')[1:]]
                    metrics_per_tool.append(int(parts[0]))
                    col = parts[2].strip()
                else:
                    row = row.replace(col, ' \\multicolumn{1}{c|}{' + col + '} ')
                    metrics_per_tool.append(1)
            # Bold/italicize
            row = row.replace(col, f'\\{styles[row_title]}' + '{' + col + '}')
            # Word wrap
            row, col = cls.word_wrap(row, col)

        # Add border
        if row_title == 'Tool':
            row += indent + '\\hline\n'
        # Check if footnotes are needed
        elif row_title == 'Metric':
            for key, footnote in cls.FOOTNOTES.items():
                if key in row:
                    footnotes.append(footnote)
        return row, metrics_per_tool, footnotes

    @classmethod
    def word_wrap(cls, row: str, col: str, max_length: int = 10):
        """Wrap long words in a LaTeX table cell.

        :param str row: The row to format.
        :param str col: The cell content.
        :param list[str] footnotes: The footnotes to add to the table.
        """
        if len(col) > max_length and ' ' in col:
            # Required markup for multi-line text
            row = row.replace(col, f'\\vtop{{\\hbox{{\\strut {col}}}}}')
            # Word wrap column name using latex instead of newline
            wrapped_text = textwrap.wrap(col, width=max_length, break_long_words=False)
            row = row.replace(col, '}\\hbox{\\strut '.join(wrapped_text))
        return row, col
