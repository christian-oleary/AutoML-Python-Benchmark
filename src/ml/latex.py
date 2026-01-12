"""Functions for LaTeX output generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series
from pylatex import MultiColumn, MultiRow, NoEscape, Tabular  # pylint: disable=W0611  # noqa: F401
from pylatex.utils import bold, escape_latex, italic
from scipy.stats import pearsonr, spearmanr
from typing_extensions import Literal


LEFT_ALIGN = '>{\\raggedleft\\arraybackslash}'
Highlight = Literal[None, 'bold', 'italic']
FontSize = Literal['normalsize', 'small', 'footnotesize', 'scriptsize']


class LatexGenerator:
    """Generates TeX for documentation."""

    precision: int | None

    def __init__(self, precision: int | None = None) -> None:
        self.precision = precision

    def format_str(self, text: str, method: Highlight, escape: bool = True) -> str:
        """Format string for LaTeX output."""
        if escape:
            text = escape_latex(text)
        if method == 'bold':
            return bold(text)
        if method == 'italic':
            return italic(text)
        return text

    def format_max_in_series(self, row: Series, **kwargs) -> Series:
        """Format the maximum numeric value in a pandas Series.

        :param pd.Series row: The input Series.
        :return pd.Series: The modified Series with the max value formatted.
        """
        precision: int | None = kwargs.get('precision')
        method: Highlight | None = kwargs.get('method')

        # Convert numeric columns, ignore non-numerics
        numeric_vals = pd.to_numeric(row, errors='coerce')
        max_val = numeric_vals.max()

        # Build new series with bold for max value, preserving column structure
        new_row = row.copy()
        for col in row.index:
            val = row[col]
            try:
                num_val = float(val)
                # Hide NaN values
                if pd.isna(num_val):
                    new_row[col] = ''
                else:
                    # Round value if precision is specified
                    if precision is not None:
                        num_val = round(num_val, precision)
                    # Format the max value
                    if num_val == max_val:
                        # new_row[col] = f'\\textbf{{{num_val}}}'
                        new_row[col] = self.format_str(
                            text=str(num_val), method=method, escape=False
                        )
                    else:
                        new_row[col] = f'{num_val}' if isinstance(val, (float, int)) else val
            except (ValueError, TypeError):
                if pd.isna(val) or val == 'nan':
                    new_row[col] = ''
                else:
                    new_row[col] = val
        return new_row

    def monospace(self, text: str, escape: bool = True) -> str:
        """Return text in monospace font for code snippets.

        :param str text: The text to format in monospace font.
        :param bool escape: Whether to escape LaTeX special characters (default is True).
        :return str: The text formatted in monospace font for LaTeX.
        """
        if escape:
            text = escape_latex(text)
        return '\\texttt{' + text + '}'

    def pivot_table(
        self,
        df: DataFrame,
        column: str,
        index: str,
        values_col: str,
        format_max_values: Highlight = None,
        drop_missing: bool = True,
        precision: int | None = None,
    ) -> DataFrame:
        """Create a pivot table from the given DataFrame and add two rows for mean and median values.

        :param DataFrame df: The input DataFrame.
        :param str column: The column to pivot on, i.e., the x-axis.
        :param str index: The index column, i.e., the y-axis.
        :param str values_col: The values to aggregate.
        :param Highlight format_max_values: Format max. values as 'bold' or 'italic' (default=None).
        :param bool drop_missing: Drop rows with missing values (default is True).
        :param int | None precision: The number of decimal places to round to (default is None).
        :return DataFrame: The pivoted and formatted DataFrame.
        """
        df = df.pivot_table(
            index=index,  # y-axis
            columns=column,  # x-axis
            values=values_col,
            aggfunc='mean',
            fill_value=None,
        )
        # Drop rows with missing values
        if drop_missing:
            df = df.dropna(how='any', axis=0)

        # Calculate mean and median values for each column
        means = df.mean(axis=0, skipna=True, numeric_only=True)
        medians = df.median(axis=0, skipna=True, numeric_only=True)

        # Round to given precision
        if precision is not None:
            if self.precision:
                precision = self.precision
            means, medians = means.round(precision), medians.round(precision)

        # Format additional rows in bold
        means, medians = means.apply(bold), medians.apply(bold)

        # Format max. value in each row ignoring non-numeric values
        df_formatted = df.apply(
            self.format_max_in_series, axis=1, precision=precision, method=format_max_values
        )
        df_formatted.columns = df.columns  # Keep original column names
        df = df_formatted.fillna('')  # Replace NaN with empty string

        # Add final row(s) for column averages in bold
        df.loc[bold(f'Mean {values_col}')] = means.apply(italic)
        df.loc[bold(f'Median {values_col}')] = medians.apply(italic)

        # Add model names as first column
        # df.insert(0, MODEL_COL, df.index)
        # df.loc[mean, MODEL_COL] = '\\hline\n' + mean + ':'
        # df.loc[median, MODEL_COL] = '\\hline\n' + median + ':'
        return df.fillna('')

    def readable_comma_list(self, items: list[str], monospace: bool = False) -> str:
        """Convert a list of items to a readable comma-separated string.

        :param list[str] items: List of items to convert.
        :param bool monospace: Whether to format items in monospace font (default is False).
        :return str: A string containing the items formatted as a comma-separated list.
        """
        # Validate inputs
        if not isinstance(items, list):
            raise TypeError(f'Expected a list of items. Got: {type(items)}')
        if len(items) == 0:
            raise ValueError('List of items is empty.')
        if not isinstance(monospace, bool):
            raise TypeError(f'Expected monospace to be a boolean. Got: {type(monospace)}')

        # Apply monospace formatting if needed
        if monospace:
            items = [self.monospace(item) for item in items]

        # Format as strings and strip
        items = [str(item).strip() for item in items]

        # One item
        if len(items) == 1:
            return items[0]
        # More than two items
        return ', '.join(items[:-1]) + ' and ' + items[-1]

    def save_text_as_tex(self, tex_str: str, output_file: Path) -> None:
        """Save a text string as a LaTeX file.

        :param str tex_str: The text to save formatted as LaTeX.
        :param Path output_file: The path to the output LaTeX file.
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open('w', encoding='utf-8') as f:
            f.write(tex_str)

    def series_correlation(
        self,
        series_1: Series,
        series_2: Series,
        alpha: float = 0.01,
        method: Literal['spearman', 'pearson'] = 'spearman',
        precision: int = 8,
    ) -> tuple[float, float, bool]:
        """Calculate correlations between values in two DataFrames.

        :param str model_name: Name of the model being compared
        :param pd.Series series_1: First series of values
        :param pd.Series series_2: Second series of values
        :param float alpha: Significance level, defaults to 0.01
        :param str method: Correlation method ('spearman' or 'pearson'), defaults to 'spearman'
        :param int precision: Number of decimal places to round the results, defaults to 8
        :return tuple[float, float, bool]: Correlation coefficient, p-value, and significance boolean
        """
        # Calculate correlation and p-value
        if method == 'spearman':
            corr, p_value = spearmanr(series_1.values, series_2.values)
        elif method == 'pearson':
            corr, p_value = pearsonr(series_1.values, series_2.values)
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
        # Round results if precision is specified
        if precision is not None:
            corr = corr.round(precision)
            p_value = p_value.round(precision)
        return corr, p_value, p_value < alpha

    def tex_table(
        self,
        df: DataFrame,
        caption: str,
        booktabs: bool = False,
        label: str | None = None,
        precision: int | None = None,
        striped_rows: bool = True,
        **kwargs,
    ):
        """Generate a LaTeX table from a DataFrame and save it to a file.

        :param DataFrame df: DataFrame containing results to convert to a LaTeX table.
        :param Path output_file: Path to save the generated LaTeX table file.
        :param str caption: Caption for the LaTeX table.
        :param bool booktabs: Whether to use booktabs style for the table (default is False).
        :param str | None label: Label for the LaTeX table, used for referencing.
        :param int | None precision: Number of decimal places to round the scores (default is None).
        :param bool striped_rows: Whether to use striped rows for better readability (default is True).
        :param kwargs: Additional keyword arguments for table formatting.
        """
        if label is None:  # Generate label from caption
            label = caption.lower().strip().replace(' ', '-').replace('.', '')

        # Create table header
        table = self._table_header(df.columns.tolist(), booktabs=booktabs, **kwargs)

        # Round decimal places
        if precision is not None:
            df = df.round(precision)

        # Add rows to the table
        count = 0
        for _, row in df.iterrows():
            color = 'gray!5' if striped_rows and count % 2 == 0 else 'white'
            table.add_row([NoEscape(cell) for cell in row.to_list()], color=color)
            count += 1
        if not booktabs:
            table.add_hline()

        # Wrap 'tabular' in 'tabular*'
        tex_str = self._wrap_tabular(table=table, caption=caption, label=label, **kwargs)
        return tex_str

    def word_wrap(self, text: str, max_characters: int = 100, sep=' ') -> list[str]:
        """Wrap text in LaTeX. Avoids splitting inside braces.

        :param str text: Input text to be wrapped.
        :param int max_characters: Maximum number of characters per line (default is 100).
        :param str sep: Separator to use for splitting the text (default is space).
        :return list[str]: List of wrapped lines, each fitting within the specified character limit.
        """
        lines = []
        current_line = ''
        braces_level = 0  # Prevent splitting inside braces by counting
        # Split the text into words and iterate through them
        for word in text.split(sep):
            # Start new line if adding the word exceeds the character limit
            if braces_level < 1 and (len(current_line) + len(word) + 1 > max_characters):
                lines.append(current_line.strip())  # Add the current line
                current_line = word + sep  # Start next line
            # if it fits, add the word to the current line
            else:
                current_line += word + sep

            # Track braces level to avoid splitting inside braces for next 'word'
            braces_level += word.count('{') - word.count('}')

        # Add the last line if it exists
        if current_line:
            lines.append(current_line.strip())
        return lines

    def _table_header(
        self,
        col_names: list,
        booktabs: bool = False,
        escape: bool = True,
        multicol_align: str = 'c|',
        split_pattern: str | None = None,
        table_spec: str | None = None,
        **_,
    ) -> Tabular:
        """Add a header row to the LaTeX table.

        :param list col_names: List of header values to add to the table.
        :param bool booktabs: Whether to use booktabs style (default is False).
        :param bool escape: Escape LaTeX special characters in column names (default is True).
        :param str multicol_align: Alignment for MultiColumn cells (default is 'c|').
        :param str | None split_pattern: Pattern to split column names into multiple rows.
        :param str | None table_spec: Column format for the table.
        :return Tabular: A LaTeX tabular environment with the specified header row.
        """
        if table_spec is None:
            table_spec = '|'.join(['l'] * len(col_names))

        if len(col_names) == 0:
            raise ValueError('Column names list is empty.')

        # Check if any column name is a list or tuple (indicating multi-index)
        multi_index = any(isinstance(col, (list, tuple)) for col in col_names)

        # Create LaTeX tabular environment with column names and format
        table = Tabular(table_spec=table_spec, booktabs=booktabs)
        if not booktabs:
            table.add_hline()

        if split_pattern is None:
            table.add_row(*col_names, escape=escape)
        else:
            if multi_index:
                self._two_level_header(
                    table=table,
                    col_names=[
                        col if isinstance(col, (list, tuple)) else [col] for col in col_names
                    ],  # type: ignore
                    borders=not booktabs,
                    escape=escape,
                    multicol_align=multicol_align,
                )
            else:
                table.add_row(*col_names, escape=escape)

        if not booktabs:
            table.add_hline()
        return table

    def _two_level_header(
        self,
        table: Tabular,
        col_names: list,
        borders: bool = True,
        escape: bool = True,
        multicol_align: str = 'c|',
        width: str = '1.5cm',
    ) -> Tabular:
        """Create a two-level header for the LaTeX table.

        :param Tabular table: The LaTeX table to add the header to.
        :param list col_names: List of column names, each split into parts.
        :param bool borders: Whether to include borders in the table (default is True).
        :param bool escape: Whether to escape LaTeX special characters (default is True).
        :param str multicol_align: Alignment for MultiColumn cells (default is 'c|').
        :param str width: Width for MultiRow cells (default is '1.5cm').
        :return Tabular: The LaTeX table with the added two-level header.
        """
        first_row, second_row = [], []

        # Build first and second row based on parts
        for headings in col_names:
            # If the column name is a single word, use MultiRow to span two rows
            if len(headings) == 1:
                first_row.append(MultiRow(2, width=width, data=headings[0]))
                second_row.append('')
            # If the column name has two parts, split into two rows
            elif len(headings) == 2:
                first_row.append(headings[0])
                second_row.append(headings[1])
            else:
                raise ValueError(f'Unexpected number of parts: {len(headings)} ({headings})')

        # Merge identical first row headers into two or more merged cells
        counts = {col: first_row.count(col) for col in first_row if isinstance(col, str)}
        dropped_cols = []
        header_borders = []
        for col, count in counts.items():
            if count > 1:
                index = first_row.index(col)
                # Replace first occurrence with MultiColumn
                first_row[index] = MultiColumn(count, data=col, align=multicol_align)
                dropped_cols.append(col)
                # Add borders in between merged cells and second row
                if borders:
                    header_borders.append((index, count))

        # Remove columns replaced with MultiColumn from the first row
        first_row = [x for x in first_row if x not in dropped_cols]

        # Add the first row of headings, borders, and second row
        table.add_row(*first_row, escape=escape)
        # Add borders
        if borders:
            for border in header_borders:
                table.append(
                    NoEscape('\\cline{' + f'{border[0] + 1}-{border[0] + border[1]}' + '}')
                )
        # Add the second row of headings
        table.add_row(*second_row, escape=escape)
        return table

    def _wrap_tabular(
        self,
        table: Tabular,
        caption: str,
        label: str,
        extra_row_height: str = '1pt',
        size: FontSize = 'small',
        suffix: str = '*',
        **_,
    ) -> str:
        """Wrap a tabular LaTeX element using table*.

        :param Tabular table: The LaTeX table to wrap.
        :param str caption: Caption for the table.
        :param str label: Label for the table.
        :param str extra_row_height: Height of the extra table rows (default is '1pt').
        :param str size: Size of the table (default is 'small').
        :param str suffix: Suffix for the table environment (default is '*').
        :return str: A string containing the LaTeX code for the wrapped table.
        """
        tabular = table.dumps().replace('&', ' & ').replace('%', ' %')
        return (
            '\\begin{table' + suffix + '}\n'
            '\\caption{' + caption + '}\n'
            '\\' + size + '\n'
            '\\centering\n'
            '\\setlength\\extrarowheight{' + extra_row_height + '}\n'
            '\\label{tab:' + label + '}\n'
            f'{tabular}\n'
            '\\end{table' + suffix + '}\n'
        )
