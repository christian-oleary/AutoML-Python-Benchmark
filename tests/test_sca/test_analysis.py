"""Tests for the sca.analysis module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from ml import Library
from ml.sca.analysis import Analysis
from ml.sca.repo import GitRepo

# pylint: disable=protected-access,redefined-outer-name,unused-argument


# DEFAULTS
DEFAULT_NAME = 'test-repo'
DEFAULT_EMAIL = 'test@example.com'
DEFAULT_URL = 'https://github.com/user/repo.git'
INPUT_PATH = 'input'
OUTPUT_PATH = 'output'


@pytest.fixture
def mock_library():
    """Create a mock Library object."""
    lib = Mock(spec=Library)
    lib.git_name = 'test-repo'
    lib.git_url = 'https://github.com/user/test-repo.git'
    lib.package_name = 'test'
    return lib


@pytest.fixture
def mock_git_repo(mock_library: Mock):
    """Fixture to return a mock GitRepo object."""

    def _create_mock_git_repo(name, url, path):
        git_repo = Mock(spec=GitRepo)
        git_repo.library = mock_library
        git_repo.library.git_name = name
        git_repo.library.git_url = url
        git_repo.path = path
        return git_repo

    return _create_mock_git_repo


@pytest.fixture
def analysis(tmp_path: Path):
    """Fixture to return an Analysis object."""
    return Analysis(input_dir=Path(tmp_path, INPUT_PATH), output_dir=Path(tmp_path, OUTPUT_PATH))


def test_analyze_repo(analysis: Analysis, tmp_path: Path, mock_library: Mock, mock_git_repo):
    """Test the analyze_repo method of the Analysis class."""
    with (
        patch.object(Analysis, '_parse_sonar_scanner_json', return_value={'lines': 1}),
        patch.object(Analysis, '_read_coverage_xml', return_value={'lines': 1}),
        patch.object(Analysis, '_git_analysis', return_value={'lines': 1}),
        patch.object(Analysis, 'build_commands', return_value={}),
        patch.object(Analysis, '_run_cli_command', return_value={}),
        patch.object(
            GitRepo,
            'from_package_name',
            return_value=mock_git_repo(DEFAULT_NAME, DEFAULT_URL, tmp_path),
        ),
    ):

        result = analysis.analyze_repo(tmp_path)
        assert result['name'] == DEFAULT_NAME
        assert str(Path(result['path'])) == str(tmp_path)
        assert 'coverage__lines' in result
        # assert 'git__lines' in result
        assert 'sonar__lines' in result


def test_parse_pylint_empty_output(analysis: Analysis):
    """Test the parse_pylint method of the Analysis class with empty output."""
    output_json = {'statistics': {'messageTypeCount': {}, 'score': 0}, 'messages': []}
    expected = {'score': 0}
    assert analysis._parse_pylint(output_json) == expected


def test_parse_pylint_single_message(analysis: Analysis):
    """Test the parse_pylint method of the Analysis class with a single message."""
    output_json = {
        'statistics': {'messageTypeCount': {}, 'score': 10},
        'messages': [{'messageId': 'C0103'}],
    }
    assert analysis._parse_pylint(output_json) == {'score': 10}


def test_parse_json_bandit_empty_output(analysis: Analysis):
    """Test the parse_json_bandit method with empty output."""
    output_json = {'metrics': {'_totals': {}}, 'results': []}
    assert analysis._parse_json_bandit(output_json) == {}


def test_parse_json_bandit_single_issue(analysis: Analysis):
    """Test the parse_json_bandit method with a single issue."""
    output_json = {'metrics': {'_totals': {'loc': 100}}, 'results': [{'test_id': 'B101'}]}
    assert analysis._parse_json_bandit(output_json) == {'loc': 100, 'B101': 1.0}


def test_parse_json_bandit_multiple_issues(analysis: Analysis):
    """Test the parse_json_bandit method with multiple issues."""
    output_json = {
        'metrics': {'_totals': {'loc': 100}},
        'results': [{'test_id': 'B101'}, {'test_id': 'B102'}],
    }
    assert analysis._parse_json_bandit(output_json) == {'loc': 100, 'B101': 1.0, 'B102': 1.0}


@pytest.mark.parametrize(
    "language_distribution, expected_percentage",
    [
        ("py=50;js=50", 0.5),
        ("py=30;js=70", 0.3),
        ("py=100", 1.0),
        ("js=100", 0.0),
        ("py=0;js=100", 0.0),
        ("py=25;js=25;java=50", 0.25),
    ],
)
def test_calculate_python_percentage(language_distribution, expected_percentage):
    """Test the calculate_python_percentage method of the Analysis class."""
    analysis = Analysis(input_dir="dummy_input", output_dir="dummy_output")
    assert analysis.calculate_python_percentage(language_distribution) == expected_percentage
