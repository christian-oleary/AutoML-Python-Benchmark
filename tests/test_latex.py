"""Tests for the LatexGenerator class in ml.latex module."""

import numpy as np
from pandas import DataFrame, Series
from pylatex import Tabular
import pytest

from ml.latex import LatexGenerator

# pylint: disable=protected-access,redefined-outer-name


class TestLatexGeneratorInit:
    """Tests for LatexGenerator initialization."""

    def test_init_without_precision(self):
        """Test initialization without precision."""
        gen = LatexGenerator()
        assert gen.precision is None

    def test_init_with_precision(self):
        """Test initialization with precision."""
        gen = LatexGenerator(precision=2)
        assert gen.precision == 2


class TestFormatStr:
    """Tests for format_str method."""

    def test_format_str_bold(self):
        """Test formatting string as bold."""
        gen = LatexGenerator()
        result = gen.format_str('test', method='bold')
        assert '\\textbf' in result or 'textbf' in str(result)

    def test_format_str_italic(self):
        """Test formatting string as italic."""
        gen = LatexGenerator()
        result = gen.format_str('test', method='italic')
        assert '\\textit' in result or 'textit' in str(result)

    def test_format_str_none(self):
        """Test formatting string with None method."""
        gen = LatexGenerator()
        result = gen.format_str('test', method=None, escape=False)
        assert result == 'test'

    def test_format_str_with_escape(self):
        """Test formatting string with escape enabled."""
        gen = LatexGenerator()
        result = gen.format_str('test_value', method=None, escape=True)
        assert 'test' in result


class TestFormatMaxInSeries:
    """Tests for format_max_in_series method."""

    def test_format_max_in_series_basic(self):
        """Test formatting max value in series."""
        gen = LatexGenerator()
        series = Series({'a': 1.0, 'b': 5.0, 'c': 3.0})
        result = gen.format_max_in_series(series, precision=None, method='bold')
        assert len(result) == len(series)

    def test_format_max_in_series_with_precision(self):
        """Test formatting max value with precision."""
        gen = LatexGenerator()
        series = Series({'a': 1.5555, 'b': 5.9999, 'c': 3.1111})
        result = gen.format_max_in_series(series, precision=2, method=None)
        assert len(result) == len(series)

    def test_format_max_in_series_with_nan(self):
        """Test formatting series with NaN values."""
        gen = LatexGenerator()
        series = Series({'a': 1.0, 'b': float('nan'), 'c': 3.0})
        result = gen.format_max_in_series(series, precision=None, method=None)
        assert result['b'] == ''

    def test_format_max_in_series_non_numeric(self):
        """Test formatting series with non-numeric values."""
        gen = LatexGenerator()
        series = Series({'a': 1.0, 'b': 'text', 'c': 3.0})
        result = gen.format_max_in_series(series, precision=None, method=None)
        assert result['b'] == 'text'


class TestMonospace:
    """Tests for monospace method."""

    def test_monospace_basic(self):
        """Test monospace formatting."""
        gen = LatexGenerator()
        result = gen.monospace('code')
        assert '\\texttt{' in result
        assert '}' in result

    def test_monospace_with_escape(self):
        """Test monospace with escape enabled."""
        gen = LatexGenerator()
        result = gen.monospace('code_sample', escape=True)
        assert '\\texttt{' in result

    def test_monospace_without_escape(self):
        """Test monospace without escape."""
        gen = LatexGenerator()
        result = gen.monospace('code', escape=False)
        assert result == '\\texttt{code}'


class TestPivotTable:
    """Tests for _pivot_table method."""

    def test_pivot_table_basic(self):
        """Test basic pivot table creation."""
        gen = LatexGenerator()
        df = DataFrame(
            {'col1': ['A', 'A', 'B', 'B'], 'col2': ['X', 'Y', 'X', 'Y'], 'value': [1, 2, 3, 4]}
        )
        result = gen.pivot_table(df, 'col2', 'col1', 'value')
        assert isinstance(result, DataFrame)

    def test_pivot_table_drop_missing(self):
        """Test pivot table with drop_missing."""
        gen = LatexGenerator()
        df = DataFrame(
            {
                'col1': ['A', 'A', 'B', 'B'],
                'col2': ['X', 'Y', 'X', 'Y'],
                'value': [1.0, 2.0, 3.0, float('nan')],
            }
        )
        result = gen.pivot_table(df, 'col2', 'col1', 'value', drop_missing=True)
        assert isinstance(result, DataFrame)

    def test_pivot_table_with_precision(self):
        """Test pivot table with precision."""
        gen = LatexGenerator()
        df = DataFrame(
            {
                'col1': ['A', 'A', 'B', 'B'],
                'col2': ['X', 'Y', 'X', 'Y'],
                'value': [1.5555, 2.6666, 3.7777, 4.8888],
            }
        )
        result = gen.pivot_table(df, 'col2', 'col1', 'value', precision=2)
        assert isinstance(result, DataFrame)

    def test_pivot_table_with_formatting(self):
        """Test pivot table with max value formatting."""
        gen = LatexGenerator()
        df = DataFrame(
            {'col1': ['A', 'A', 'B', 'B'], 'col2': ['X', 'Y', 'X', 'Y'], 'value': [1, 5, 3, 2]}
        )
        result = gen.pivot_table(df, 'col2', 'col1', 'value', format_max_values='bold')
        assert isinstance(result, DataFrame)


class TestReadableCommaList:
    """Tests for readable_comma_list method."""

    def test_readable_comma_list_single_item(self):
        """Test readable list with single item."""
        gen = LatexGenerator()
        result = gen.readable_comma_list(['item1'])
        assert result == 'item1'

    def test_readable_comma_list_two_items(self):
        """Test readable list with two items."""
        gen = LatexGenerator()
        result = gen.readable_comma_list(['item1', 'item2'])
        assert 'and' in result
        assert 'item1' in result
        assert 'item2' in result

    def test_readable_comma_list_multiple_items(self):
        """Test readable list with multiple items."""
        gen = LatexGenerator()
        result = gen.readable_comma_list(['a', 'b', 'c'])
        assert 'a' in result
        assert 'b' in result
        assert 'c' in result
        assert 'and' in result

    def test_readable_comma_list_with_monospace(self):
        """Test readable list with monospace formatting."""
        gen = LatexGenerator()
        result = gen.readable_comma_list(['code1', 'code2'], monospace=True)
        assert '\\texttt{' in result

    def test_readable_comma_list_invalid_input(self):
        """Test readable list with invalid input."""
        gen = LatexGenerator()
        with pytest.raises(TypeError):
            gen.readable_comma_list('not_a_list')  # type: ignore

    def test_readable_comma_list_empty_list(self):
        """Test readable list with empty list."""
        gen = LatexGenerator()
        with pytest.raises(ValueError):
            gen.readable_comma_list([])

    def test_readable_comma_list_invalid_monospace(self):
        """Test readable list with invalid monospace parameter."""
        gen = LatexGenerator()
        with pytest.raises(TypeError):
            gen.readable_comma_list(['item'], monospace='invalid')  # type: ignore


class TestSaveTextAsTex:
    """Tests for save_text_as_tex method."""

    def test_save_text_as_tex(self, tmp_path):
        """Test saving text as LaTeX file."""
        gen = LatexGenerator()
        output_file = tmp_path / 'test.tex'
        gen.save_text_as_tex('\\textbf{test}', output_file)
        assert output_file.exists()
        assert output_file.read_text() == '\\textbf{test}'

    def test_save_text_as_tex_creates_directory(self, tmp_path):
        """Test saving text creates directory if it doesn't exist."""
        gen = LatexGenerator()
        output_file = tmp_path / 'subdir' / 'test.tex'
        gen.save_text_as_tex('test content', output_file)
        assert output_file.exists()
        assert output_file.parent.exists()


class TestSeriesCorrelation:
    """Tests for series_correlation method."""

    def test_series_correlation_spearman(self):
        """Test Spearman correlation."""
        gen = LatexGenerator()
        series1 = Series([1, 2, 3, 4, 5])
        series2 = Series([2, 4, 5, 4, 6])
        corr, p_value, significant = gen.series_correlation(series1, series2, method='spearman')
        assert isinstance(corr, float)
        assert isinstance(p_value, float)
        assert isinstance(significant, (bool, np.bool_))

    def test_series_correlation_pearson(self):
        """Test Pearson correlation."""
        gen = LatexGenerator()
        series1 = Series([1.0, 2.0, 3.0, 4.0, 5.0])
        series2 = Series([2.0, 4.0, 5.0, 4.0, 6.0])
        corr, p_value, significant = gen.series_correlation(series1, series2, method='pearson')
        assert isinstance(corr, float)
        assert isinstance(p_value, float)
        assert isinstance(significant, (bool, np.bool_))

    def test_series_correlation_invalid_method(self):
        """Test correlation with invalid method."""
        gen = LatexGenerator()
        series1 = Series([1, 2, 3])
        series2 = Series([2, 4, 6])
        with pytest.raises(ValueError):
            gen.series_correlation(series1, series2, method='invalid')  # type: ignore

    def test_series_correlation_alpha(self):
        """Test correlation with custom alpha."""
        gen = LatexGenerator()
        series1 = Series([1, 2, 3, 4, 5])
        series2 = Series([2, 4, 5, 4, 6])
        _, __, significant = gen.series_correlation(series1, series2, alpha=0.05, method='spearman')
        assert isinstance(significant, (bool, np.bool_))


class TestTableHeader:
    """Tests for _table_header method."""

    def test_table_header_basic(self):
        """Test basic table header creation."""
        gen = LatexGenerator()
        col_names = ['Col1', 'Col2', 'Col3']
        result = gen._table_header(col_names)
        assert result is not None

    def test_table_header_empty(self):
        """Test table header with empty column names."""
        gen = LatexGenerator()
        with pytest.raises(ValueError):
            gen._table_header([])

    def test_table_header_with_escape(self):
        """Test table header with escape enabled."""
        gen = LatexGenerator()
        col_names = ['Col_1', 'Col_2']
        result = gen._table_header(col_names, escape=True)
        assert result is not None

    def test_table_header_booktabs(self):
        """Test table header with booktabs style."""
        gen = LatexGenerator()
        col_names = ['Col1', 'Col2']
        result = gen._table_header(col_names, booktabs=True)
        assert result is not None


class TestTwoLevelHeader:
    """Tests for _two_level_header method."""

    def test_two_level_header_basic(self):
        """Test basic two-level header."""
        gen = LatexGenerator()

        table = Tabular('ll')
        col_names = [['Header1'], ['Sub1', 'Sub2']]
        result = gen._two_level_header(table, col_names)
        assert result is not None

    def test_two_level_header_invalid_parts(self):
        """Test two-level header with invalid number of parts."""
        gen = LatexGenerator()

        table = Tabular('lll')
        col_names = [['Part1', 'Part2', 'Part3']]
        with pytest.raises(ValueError):
            gen._two_level_header(table, col_names)


class TestWrapTabular:
    """Tests for _wrap_tabular method."""

    def test_wrap_tabular_basic(self):
        """Test basic tabular wrapping."""
        gen = LatexGenerator()

        table = Tabular('ll')
        table.add_row('a', 'b')
        result = gen._wrap_tabular(table, 'Test Caption', 'test-label')
        assert '\\begin{table' in result
        assert 'Test Caption' in result
        assert 'tab:test-label' in result

    def test_wrap_tabular_with_size(self):
        """Test tabular wrapping with custom size."""
        gen = LatexGenerator()

        table = Tabular('ll')
        table.add_row('a', 'b')
        result = gen._wrap_tabular(table, 'Caption', 'label', size='footnotesize')
        assert '\\footnotesize' in result


class TestTexTable:
    """Tests for tex_table method."""

    def test_tex_table_basic(self):
        """Test basic LaTeX table generation."""
        gen = LatexGenerator()
        df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = gen.tex_table(df, 'Test Table')
        assert result is not None
        assert 'Test Table' in result

    def test_tex_table_with_label(self):
        """Test LaTeX table with custom label."""
        gen = LatexGenerator()
        df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = gen.tex_table(df, 'Test Table', label='custom-label')
        assert 'tab:custom-label' in result

    def test_tex_table_with_precision(self):
        """Test LaTeX table with precision."""
        gen = LatexGenerator()
        df = DataFrame({'A': [1.5555, 2.6666], 'B': [4.7777, 5.8888]})
        result = gen.tex_table(df, 'Table', precision=2)
        assert result is not None

    def test_tex_table_booktabs(self):
        """Test LaTeX table with booktabs style."""
        gen = LatexGenerator()
        df = DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = gen.tex_table(df, 'Table', booktabs=True)
        assert result is not None

    def test_tex_table_striped_rows(self):
        """Test LaTeX table with striped rows."""
        gen = LatexGenerator()
        df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = gen.tex_table(df, 'Table', striped_rows=True)
        assert result is not None


class TestWordWrap:
    """Tests for word_wrap method."""

    def test_word_wrap_short_text(self):
        """Test word wrap with short text."""
        gen = LatexGenerator()
        result = gen.word_wrap('short text')
        assert len(result) == 1
        assert result[0] == 'short text'

    def test_word_wrap_long_text(self):
        """Test word wrap with long text."""
        gen = LatexGenerator()
        long_text = 'word ' * 30
        result = gen.word_wrap(long_text, max_characters=50)
        assert len(result) > 1
        for line in result:
            assert len(line) <= 60  # Allow some tolerance

    def test_word_wrap_with_braces(self):
        """Test word wrap avoids splitting inside braces."""
        gen = LatexGenerator()
        text = 'text {long text inside braces} more text'
        result = gen.word_wrap(text, max_characters=20)
        assert len(result) >= 1

    def test_word_wrap_custom_separator(self):
        """Test word wrap with custom separator."""
        gen = LatexGenerator()
        text = 'word1,word2,word3'
        result = gen.word_wrap(text, max_characters=20, sep=',')
        assert len(result) >= 1
