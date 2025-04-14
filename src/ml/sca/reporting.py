"""Functionality for reporting the results of source code analysis."""

import json
import sys
import textwrap
from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

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
        'LoC': ITALIC + '{ LoC: Lines of Code}',
        ' MI': ITALIC + '{ MI: Maintainability Index}',
        'NCLoC': ITALIC + '{ NCLoC: Non-Comment Lines of Code}',
        'SLoC': ITALIC + '{ SLoC: Source Lines of Code}',
        # 'SQALE': ITALIC + '{ SQALE: Software Quality Assessment based on Lifecycle Expectations}',
    }
    INTEGER_PATTERNS = [  # Column names patterns to convert to integers
        'Cognitive Complexity',
        'Comments',
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
        'Sonar: Complexity',
        'Smells',
        'spots',
        'SQALE Index',
        'Violations',
    ]
    LONGEST_NAME = max(len(name) for name in LIBRARIES.values())
    # fmt: off
    RANK_METRICS = [
        # ##### Bandit and Coverage #####
        'bandit__Total Issues', 'bandit__Issues per Line', 'coverage__line-rate',
        # ##### Pylint, Ruff and Prospector #####
        'pylint__score', 'ruff__Ruff Score', 'Prospector Num. Issues',
        # ##### Radon #####
        'radon-cc__mean-cc_class',  # Code complexity
        'radon-cc__mean-cc_function',
        'radon-cc__mean-cc_method',
        'radon-hal__Mean Halstead Bugs',  # Risk estimation
        'radon-hal__Mean Halstead Effort',  # Dev difficulty & maintainability
        'radon-hal__Mean Halstead Volume',  # Code size & complexity
        'radon-mi__Mean Maintainability Index',  # Maintainability
        # 'radon-raw__LLoC', 'radon-raw__SLoC',  # Alternative LoC metrics added later
        # radon-raw__Blank Lines, # radon-raw__Comments, # radon-raw__LoC,
        # radon-raw__Multi-Line String Lines, # radon-raw__Single Line Comments,
        # ##### SonarQube #####
        'sonar__bugs',
        'sonar__code_smells',
        'sonar__cognitive_complexity',
        # 'sonar__comment_lines', 'sonar__comment_lines_density',
        'sonar__complexity',
        'sonar__coverage',
        'sonar__development_cost',
        'sonar__duplicated_blocks',
        'sonar__duplicated_files',
        'sonar__duplicated_lines',
        'sonar__duplicated_lines_density',
        'sonar__file_complexity',
        # sonar__ncloc, sonar__ncloc_language_distribution,
        'sonar__reliability_rating',
        'sonar__reliability_remediation_effort',
        'sonar__security_hotspots',
        'sonar__sqale_debt_ratio',
        'sonar__sqale_index',
        'sonar__uncovered_lines',
        'sonar__violations',
        # 'sonar__blocker_violations', 'sonar__critical_violations',
        # 'sonar__major_violations', 'sonar__minor_violations',
    ]
    # fmt: on
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
        used_cols, _ = cls.summarize_tools(df_summary, keys_by_tool, output_dir)
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
        df_summary, df_loc, df_coverage, df_ranks = cls.create_plots(df_summary, output_dir)

        # Comment on results as LaTeX
        # cls.write_comments(df_summary, df_loc, df_coverage, df_ranks, output_dir)
        return df_summary

    @classmethod
    def filter_rename_columns(cls, df: pd.DataFrame, ignored_keys_by_tool: dict[str, list[str]]):
        """Filter and rename the columns of the results dataframe for analysis.

        :param pd.DataFrame df: The results as a pandas DataFrame.
        :param dict[str, list[str]] ignored_keys_by_tool: Ignored keys by tool.
        :return tuple: Dropped columns, formatted DataFrame, and grouped metrics.
        """
        df_summary = df[sorted(df.columns)].copy()  # Sort columns
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
        metric_keys = [key for key in keys_by_tool['bandit'] if key.startswith('bandit__B')]
        df_summary['bandit__Total Issues'] = df_summary[metric_keys].sum(axis=1)
        df_summary['bandit__Issues per Line'] = (  # Bandit score per line
            df_summary['bandit__Total Issues'] / df_summary['bandit__loc']
        )
        dropped += keys_by_tool['bandit']  # Keep track of keys to drop

        # Coverage: drop redundant coverage metrics
        dropped += [f'coverage__{key}' for key in ignored_keys_by_tool['coverage']]
        # Coverage: difference between Sonar and Coverage.py
        # df_summary['Difference__Difference (\\%)'] = (
        #     df_summary['sonar__coverage'] - (df_summary['coverage__line-rate'] * 100)
        # )
        # keys_by_tool['Difference'] = ['Difference__Difference (\\%)']

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

        # Radon CC: only keep sum and mean
        dropped += [k for k in keys_by_tool['radon-cc'] if 'total-' not in k and 'mean-' not in k]

        # Ruff: only keep score
        df_summary['ruff__Ruff Score'] = df_summary['ruff__ruff Score']
        dropped += keys_by_tool['ruff']

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
        df_summary = df_summary[sorted(df_summary.columns)]
        return dropped, df_summary, keys_by_tool

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
                'coverage__complexity',
                'radon-cc__mean-cc_class', 'radon-cc__mean-cc_function', 'radon-cc__mean-cc_method',
                # 'radon-cc__total-cc_class', 'radon-cc__total-cc_function', 'radon-cc__total-cc_method',
                'sonar__cognitive_complexity', 'sonar__complexity', 'sonar__file_complexity',
            ],
            'Size (Radon)': [
                # 'git__LoC',  # Git (redundant with Radon and Sonar)
                'radon-mi__Num. Files', 'radon-raw__LLoC', 'radon-raw__LoC', 'radon-raw__SLoC',
                'radon-raw__Multi-Line String Lines', 'radon-raw__Single Line Comments',
                'radon-raw__Comments', 'radon-raw__Blank Lines',
            ],
            'Size (Sonar)': [
                'sonar__classes', 'sonar__comment_lines', 'sonar__comment_lines_density',
                'sonar__files', 'sonar__functions', 'sonar__lines', 'sonar__ncloc', 'sonar__statements',
            ],
            'coverage': [col for col in df.columns if 'cover' in col],  # + ['Difference__Difference (\\%)'],
            # 'Coverage (Sonar)': [col for col in df.columns if 'cover' in col and 'sonar' in col],
            # 'Coverage (Coverage.py)': [col for col in df.columns if 'cover' in col and 'sonar' not in col],
            # 'duplication': [col for col in df.columns if 'duplicat' in col],
            # 'git': ['git__Num. Commits', 'git__Num. Contributors'],
            'halstead': [
                f'radon-hal__Mean Halstead {c}' for c in ['Bugs', 'Difficulty', 'Time', 'Volume']
            ],
            # 'halstead_mean': [col for col in df.columns if 'Mean Halstead' in col],
            # 'halstead_total': [col for col in df.columns if 'Total Halstead' in col],
            'linting': [
                *[k for k in keys_by_tool['prospector'] if k.endswith('Issues')],  # Prospector
                'pylint__score', 'ruff__Ruff Score',  # Pylint, Ruff
                'sonar__errors', 'sonar__violations',  # Sonar
            ],
            'maintainability': [
                'radon-mi__Mean Maintainability Index',
                'sonar__code_smells', 'sonar__development_cost',
                'sonar__effort_to_reach_maintainability_rating_a', 'sonar__sqale_debt_ratio',
                'sonar__sqale_index', 'sonar__sqale_rating',
            ],
            # 'readability': ['radon-hal__Mean Halstead Volume', 'radon-raw__LoC', 'radon-raw__Comments'],
            'reliability': [
                'sonar__bugs', 'sonar__reliability_rating', 'sonar__reliability_remediation_effort'
            ],
            'security': [
                'bandit__Total Issues', 'bandit__Issues per Line', 'sonar__security_hotspots'
            ],
            'Sonar - Duplications': keys_by_tool['Sonar - Duplications'],
        }
        # Combine some tables
        groups['coverage and duplication'] = [*groups['coverage'], *groups['Sonar - Duplications']]
        groups['maintainability reliability and security'] = [
            *groups['maintainability'], *groups['reliability'], *groups['security']
        ]
        # for key in ['coverage', 'duplication', 'maintainability', 'reliability', 'security']:
        #     del groups[key]

        # fmt: on
        used_cols: list[str] = []
        for category, metrics in groups.items():
            try:
                # Filter columns
                df_by_tool = df[[m for m in metrics if m in df.columns]].copy()

                # Sort columns by tool
                df_by_tool = df_by_tool[sorted(df_by_tool.columns)]
                used_cols += df_by_tool.columns.to_list()

                # Save as CSV and LaTeX
                csv_path = Path(output_dir, 'csv', f'summary_{category}.csv')
                tex_path = Path(output_dir, 'tables', f'summary_{category}.tex')
                cls.save_csv_and_tex(df_by_tool, csv_path, tex_path, category)
            except KeyError as e:
                logger.error(f'{e} \ncolumns: {df.columns} \n{category}: {metrics}')
                sys.exit(1)

        # Save as CSV and LaTeX
        cls.save_csv_and_tex(
            df.copy(),
            Path(output_dir, 'csv', 'summary.csv'),
            Path(output_dir, 'tables', 'summary.tex'),
        )
        return used_cols, groups

    @classmethod
    def save_csv_and_tex(
        cls, df: pd.DataFrame, csv_path: Path, tex_path: Path, category: str | None = None
    ):
        """Save the DataFrame as a CSV and LaTeX file.

        :param pd.DataFrame df: The DataFrame being processed.
        :param str | Path csv_path: The file path to save the DataFrame as a CSV and LaTeX file.
        :param str | Path tex_path: The file path to save the DataFrame as a LaTeX file.
        :param str | None category: The category of the metrics.
        """
        # Format column names and numeric values
        # if category == 'coverage':
        #     logger.debug(df['Difference__Difference (\\%)'])
        df.columns = [cls.format_col_name(c) for c in df.columns.tolist()]
        df = cls.format_numeric_columns(df)

        # Save as CSV
        df.to_csv(csv_path.with_suffix('.csv'), index=True)

        # Multi-index columns
        split_keys = [c.split(': ') for c in df.columns]
        df.columns = pd.MultiIndex.from_tuples(split_keys, names=['Tool', 'Metric'])

        # Pad first column (library names) to align columns
        df.index = df.index.map(lambda library: library.ljust(cls.LONGEST_NAME))

        # Specify LaTeX table caption and label
        if category:
            caption = category.replace('_', ' ').title() + ' metrics for AutoML libraries'
        else:
            caption = tex_path.stem
        # column_format = None  # default format
        column_format = cls.create_column_format(df)
        label = f'tab:r:{tex_path.stem}'

        # Save as LaTeX
        df.style.format(precision=2, thousands='\\,', na_rep='-').to_latex(
            tex_path,
            column_format=column_format,
            caption=caption,
            label=label,
            hrules=True,
            multicol_align='r',
            position='!htbp',
        )
        # Format the LaTeX table for readability and consistency
        if category:
            cls.format_tex_table(df, column_format, tex_path)

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
                if 'num. issues' in col.lower():
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
        parts = column.replace('_', ' ').split('  ')
        if len(parts) > 2:
            raise ValueError(f'Invalid column name "{column}" ({len(parts)} parts)')

        # Format tool name
        tool = (
            parts[0]
            .title()
            .replace('-Raw', '')  # (Statistics)
            .replace('-Hal', ' (Halstead)')
            .replace('-Mi', '')
            .replace('-Cc', ' (Average CC)')
        )
        # Format metric name
        metric = parts[-1].title() if parts[-1] == parts[-1].lower() else parts[-1]
        metric = (
            metric.replace('Mean-Cc', 'By')
            .replace(' Halstead', '')
            .replace('Sqale', 'SQALE')
            .replace('Ncloc', 'NCLoC')
            .replace('Maintainability Index', 'MI')
            .replace('Reliability Remediation', 'Remediation')
        )
        # Special cases
        if tool == 'Coverage':
            metric = metric.replace('-', ' ')
        if metric in ['Coverage', 'Line Rate']:
            metric = 'Coverage (\\%)'

        if tool == metric and tool.split()[-2:] == ['Num.', 'Issues']:
            tool = tool.split()[0]
            metric = 'Num. Issues'

        return f'{tool}: {metric}'

    @classmethod
    def create_column_format(cls, df: pd.DataFrame) -> str:
        """Create the column format for the LaTeX table.

        :param pd.DataFrame df: The DataFrame being processed.
        :return str: The column format for the LaTeX table.
        """
        column_format = 'r|'
        previous_tool = None
        for tool, metric in df.columns:
            if previous_tool and tool != previous_tool:
                column_format += '|'
            previous_tool = tool
            # Calculate column width based on longest word in metric name
            longest = max([len(word) for word in metric.split()])
            width = str(round(0.19 + (0.13 * longest), 3))
            # Set width and align to the right
            column_format += '\n\t\t>{\\raggedleft\\arraybackslash}m{' + width + 'cm}'
        return column_format + '\n\t'

    @classmethod
    def format_tex_table(cls, df: pd.DataFrame, column_format: str, tex_path: Path):
        """Format the LaTeX table for readability and consistency.

        :param pd.DataFrame df: The DataFrame being processes.
        :param str column_format: The column format for the LaTeX table.
        :param Path tex_path: The path to the LaTeX table file.
        """
        # Additional TEX formatting
        with open(tex_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Drop index heading row
        lines = [line for line in lines if not line.startswith('name &  & ')]

        # Use table* for wide tables
        if len(df.columns) > 5:
            lines[0] = lines[0].replace('\\begin{table}', '\\begin{table*}')
            lines[-1] = lines[-1].replace('\\end{table}', '\\end{table*}')

        # Format table row by row
        lines, footnotes, end = cls.format_rows(df, lines, column_format)

        # Add footnotes row if needed
        if len(footnotes) > 0:
            row = [
                '\t' + cls.BORDER,
                '\\multicolumn{' + str(df.shape[1] + 1) + '}{c}{',
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
    def format_rows(
        cls, df: pd.DataFrame, lines: list[str], column_format: str | None = None
    ) -> tuple:
        """Format the rows of the LaTeX table for readability and consistency.

        :param pd.DataFrame df: The DataFrame being processes.
        :param list[str] lines: The lines of the LaTeX table.
        :param str | None column_format: The column format for the LaTeX table.
        :return tuple: Formatted lines, footnotes, and the index of the end of tabular
        """
        footnotes: list[str] = []
        metrics_per_tool: list[int] = []  # i.e. where to add vertical borders
        index_end = -2  # \end{tabular}
        indent = ''
        col_counter = 0  # Index of column name to be added as a comment
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
                    lines[i], metrics_per_tool, indent, footnotes, column_format
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
        cls,
        row: str,
        metrics_per_tool: list[int],
        indent: str,
        footnotes: list[str],
        column_format: str | None = None,
    ) -> tuple:
        """Format the header rows of the LaTeX table.

        :param str row: The row to format.
        :param list[int] metrics_per_tool: The number of metrics per tool.
        :param str indent: The current indentation level.
        :param list[str] footnotes: The footnotes to add to the table.
        :param str | None column_format: The column format for the LaTeX table.
        :return tuple: The formatted row and the updated metrics per tool.
        """
        # Ignore body rows
        row_title = row.split('&')[0].strip()
        if row_title not in ['Tool', 'Metric']:
            return row, metrics_per_tool, footnotes

        # Format the header row title (first cell on left)
        styles = {'Tool': cls.BOLD, 'Metric': cls.ITALIC}
        row = row.replace(
            f'{row_title} & ',  # right-align, add vertical border and bold/italicize
            '\\multicolumn{1}{r|}{' + styles[row_title] + '{' + row_title + '}} & ',
            #                |r|
        )
        # Format remaining header row cells
        # Center-align, add vertical borders
        num_tools = row.count('{r}')
        row = row.replace('{r}', '{c|}', num_tools - 1).replace('{r}', '{c}')

        # Column names in row: center-align, bold/italicize, word wrap
        cols = row.split('&')[1:]
        cols.sort(key=len, reverse=True)  # longest first
        for col in cols:
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
            row = row.replace(col, styles[row_title] + '{' + col + '}')
            # Fix error
            row = row.replace('File \\textit{Complexity}', '\\textit{File Complexity}')

            # Apply word wrapping
            row, col = cls.word_wrap(row, col, column_format)

        # Add border
        if row_title == 'Tool':
            row += indent + cls.BORDER + '\n'

        # Check if footnotes are needed
        for key, footnote in cls.FOOTNOTES.items():
            if key.lower() in row.lower():
                footnotes.append(footnote)
        return row, metrics_per_tool, footnotes

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
        :param str | Path output_dir: Path to save the analysis.
        :return pd.DataFrame: The results as a pandas DataFrame.
        """
        # Bar plot of LoC
        df_loc = df_summary[['radon-raw__LoC', 'sonar__lines']]  # 'git__LoC',
        df_loc.columns = ['Radon', 'Sonar']  # 'Git'
        cls.plot_libraries(df_loc, 'barh', 'Lines of Code (LoC)', output_dir)

        # Bar plot of 'coverage__line-rate' and 'sonar__coverage'
        df_coverage = df_summary[['coverage__line-rate', 'sonar__coverage']]
        df_coverage['coverage__line-rate'] *= 100  # Convert to percentage
        df_coverage.columns = ['Coverage.py', 'SonarQube']
        cls.plot_libraries(df_coverage, 'barh', 'Reported Coverage (%)', output_dir)

        # Invert some scores so that higher values indicate 'worse' results
        df_ranks = df_summary[cls.RANK_METRICS]
        for col in [
            'coverage__line-rate',
            'pylint__score',
            'radon-mi__Mean Maintainability Index',
            'sonar__coverage',
        ]:
            df_ranks[col] = df_ranks[col].apply(lambda x: -x if x > 0 else x)

        # Calculate library ranks across all metrics
        kwargs = {'ascending': False, 'method': 'average', 'numeric_only': True}
        df_ranks = df_ranks.rank(**kwargs)
        df_ranks['Mean Rank'] = df_ranks.mean(axis=1)

        # Calculate library ranks for Sonar metrics only
        sonar_metrics = [col for col in df_ranks.columns if 'sonar' in col]
        df_ranks['Mean Rank (Sonar Only)'] = df_ranks[sonar_metrics].rank(**kwargs).mean(axis=1)

        # Calculate library ranks for non-Sonar metrics
        non_sonar_metrics = [col for col in df_ranks.columns if col not in sonar_metrics]
        df_ranks['Mean Rank (Non-Sonar)'] = df_ranks[non_sonar_metrics].rank(**kwargs).mean(axis=1)

        # Sort by mean rank and save to CSV
        df_ranks.sort_values(by='Mean Rank', inplace=True)
        df_ranks.to_csv(Path(output_dir, 'csv', 'ranks.csv'))

        # Box plot of ranks
        df_melted = (
            df_ranks[cls.RANK_METRICS]
            .reset_index()
            .rename(columns={'name': 'Library'})
            .melt(id_vars=['Library'], var_name='Mean Rank')
        )
        cls.plot_libraries(df_melted, 'box', 'Rank achieved for each metric', output_dir, (8, 8))

        # Plot spearman rank correlation matrix
        df_ranks.corr(method='spearman').to_csv(Path(output_dir, 'correlation_rank.csv'))

        # Plot spearman rank correlation between rank and size
        df_summary['Mean Rank'] = df_ranks['Mean Rank']
        # fmt: off
        size_metrics = [
            'Mean Rank',
            'radon-raw__Comments', 'radon-raw__Blank Lines', 'radon-raw__LLoC', 'radon-raw__LoC',
            'radon-raw__SLoC', 'radon-raw__Multi-Line String Lines', 'radon-raw__Single Line Comments',
            'sonar__ncloc', 'sonar__lines', 'sonar__comment_lines_density', 'sonar__statements',
        ]
        # fmt: on
        df_summary[size_metrics].corr(method='spearman').to_csv(
            Path(output_dir, 'correlation_size.csv')
        )

        return df_summary, df_loc, df_coverage, df_ranks

    @classmethod
    def plot_libraries(
        cls,
        data: pd.DataFrame,
        kind: str,
        metric: str,
        output_dir: str | Path,
        figsize: tuple[int, int] = (8, 8),
        fontsize: int = 12,
    ):
        """Plot the results of the SCA analysis.

        :param pd.DataFrame | pd.Series data: The data to plot.
        :param str kind: The kind of plot to create.
        :param str metric: Metric used to label the plot.
        :param str | Path output_dir: Path to save the analysis.
        :param int fontsize: The font size of the plot.
        """
        # Create plots directory
        plot_dir = Path(output_dir, 'plots')
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Set colorblind-friendly style
        plt.style.use('tableau-colorblind10')

        # Create plot
        if kind == 'box':  # Seaborn box plot
            sns.boxplot(data=data, x='value', y='Library', orient='h', linewidth=1, hue='Library')
        else:  # Bar plot
            kwargs = {} if kind == 'box' else {'edgecolor': 'black', 'linewidth': 1}
            data.plot(
                kind=kind,  # type: ignore
                figsize=figsize,
                title=f'AutoML library results for {metric}',
                fontsize=fontsize,
                colormap='cividis',
                **kwargs,
            )
        plt.xlabel(metric, fontsize=fontsize)
        plt.ylabel('Library', fontsize=fontsize)

        plt.tight_layout()
        plt.savefig(plot_dir / f'{metric}.svg')
        # plt.show()
        plt.cla()
        plt.clf()
        plt.close('all')

    # @classmethod
    # def write_comments(
    #     cls,
    #     df_summary: pd.DataFrame,
    #     df_loc: pd.DataFrame,
    #     df_coverage: pd.DataFrame,
    #     df_ranks: pd.DataFrame,
    #     output_dir: str | Path
    # ):
    #     """Write comments on the results of the SCA analysis as LaTeX.

    #     :param pd.DataFrame df: The results as a pandas DataFrame.
    #     :param str | Path output_dir: Path to save the analysis.
    #     """
    #     # Create comments directory
    #     comments_dir = Path(output_dir, 'tex', 'comments')
    #     comments_dir.mkdir(parents=True, exist_ok=True)
    #     logger.info(f'Saving comments to {comments_dir}')

    #     # Compare reported coverage differences between Coverage.py and SonarQube
    #     coverage_diff = df_coverage['SonarQube'] - df_coverage['Coverage.py']
    #     coverage_diff = coverage_diff[coverage_diff != 0]
    #     coverage_diff = coverage_diff.sort_values(ascending=False)
    #     coverage_diff = coverage_diff.to_frame().reset_index()
    #     coverage_diff.columns = ['Library', 'Difference']
    #     coverage_diff.to_latex(
    #         comments_dir / 'coverage_diff.tex',
    #         caption='Difference in reported coverage between SonarQube and Coverage.py',
    #         label='tab:r:coverage_diff',
    #         index=False,
    #     )
    #     logger.debug(f'Saved coverage difference to {comments_dir / "coverage_diff.tex"}')

    #     # Write comments on the results
    #     with open(comments_dir / 'comments.tex', 'w', encoding='utf-8') as f:
    #         comments = (
    #             f'As reported in \\ref{{tab:r:coverage_diff}}, the mean difference in coverage '
    #             f'between SonarQube and Coverage.py is {coverage_diff["Difference"].mean():.2f}%.'
    #         )
    #         f.write(comments)
