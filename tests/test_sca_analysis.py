"""Tests for the sca.analysis module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ml.sca.analysis import Analysis, GitRepo

# pylint: disable=redefined-outer-name,unused-argument

# DEFAULTS
DEFAULT_BRANCHES = ['main']
DEFAULT_COMMIT = 'latest_commit'
DEFAULT_EMAIL = 'test@example.com'
DEFAULT_NAME = 'repo'
DEFAULT_URL = 'https://github.com/user/repo.git'
INPUT_PATH = 'input'
OUTPUT_PATH = 'output'


@pytest.fixture
def mock_repo():
    """Mock a Git repository."""
    with patch('ml.sca.analysis.Repo') as mock_repo:
        mock_repo = mock_repo.return_value
        mock_repo.iter_commits.return_value = [MagicMock(author=MagicMock(email=DEFAULT_EMAIL))]
        mock_repo.branches = DEFAULT_BRANCHES
        mock_repo.head.commit = DEFAULT_COMMIT
        mock_repo.remotes.origin.url = DEFAULT_URL
        yield mock_repo


@pytest.fixture
def git_repo(tmp_path, mock_repo):
    """Fixture to return a GitRepo object."""
    return GitRepo(repo_path=tmp_path)


def test_git_repo_initialization(tmp_path, git_repo):
    """Test the initialization of a GitRepo object."""
    assert git_repo.path == tmp_path
    assert git_repo.commit_count == 1
    assert git_repo.contributors == [DEFAULT_EMAIL]
    assert git_repo.lines_of_code == 0
    assert git_repo.name == DEFAULT_NAME
    assert git_repo.num_contributors == 1
    assert git_repo.url == DEFAULT_URL


def test_git_repo_get_branches(git_repo):
    """Test the get_branches method of the GitRepo class."""
    assert git_repo.get_branches() == DEFAULT_BRANCHES


def test_git_repo_get_commit_count(git_repo):
    """Test the get_commit_count method of the GitRepo class."""
    assert git_repo.get_commit_count() == 1


def test_git_repo_get_contributors(git_repo):
    """Test the get_contributors method of the GitRepo class."""
    assert git_repo.get_contributors() == [DEFAULT_EMAIL]


def test_git_repo_get_latest_commit(git_repo):
    """Test the get_latest_commit method of the GitRepo class."""
    assert git_repo.get_latest_commit() == DEFAULT_COMMIT


def test_git_repo_get_repo_name(git_repo):
    """Test the get_repo_name method of the GitRepo class."""
    assert git_repo.get_repo_name() == DEFAULT_NAME


def test_git_repo_get_lines_of_code(git_repo):
    """Test the get_lines_of_code method of the GitRepo class."""
    with patch('os.walk') as mock_walk, patch('builtins.open', new_callable=MagicMock):
        mock_walk.return_value = [('root', [], ['file.py'])]
        git_repo.get_lines_of_code()
        assert git_repo.lines_of_code == 0


def test_git_repo_get_repo_url(git_repo):
    """Test the get_repo_url method of the GitRepo class."""
    assert git_repo.get_repo_url() == DEFAULT_URL


def test_git_repo_count_contributions(git_repo):
    """Test the count_contributions method of the GitRepo class."""
    assert git_repo.count_contributions(DEFAULT_EMAIL) == 1


def test_git_repo_count_contributors(git_repo):
    """Test the count_contributors method of the GitRepo class."""
    assert git_repo.count_contributors() == 1


@pytest.fixture
def analysis(tmp_path):
    """Fixture to return an Analysis object."""
    return Analysis(input_dir=Path(tmp_path, INPUT_PATH), output_dir=Path(tmp_path, OUTPUT_PATH))


def test_analyze_repo(analysis, git_repo, tmp_path):
    """Test the analyze_repo method of the Analysis class."""
    with (
        patch.object(Analysis, 'parse_sonar_scanner_json', return_value={'lines': 1}),
        patch.object(Analysis, 'read_coverage_xml', return_value={'lines': 1}),
        patch.object(Analysis, 'git_analysis', return_value={'lines': 1}),
        patch.object(Analysis, 'build_commands', return_value={}),
        patch.object(Analysis, 'run_cli_command', return_value={}),
    ):

        result = analysis.analyze_repo(tmp_path)
        assert result['name'] == DEFAULT_NAME
        assert str(Path(result['path'])) == str(tmp_path)
        assert 'coverage__lines' in result
        assert 'git__lines' in result
        assert 'sonar__lines' in result


def test_parse_pylint_empty_output(analysis):
    """Test the parse_pylint method of the Analysis class with empty output."""
    output_json = {'statistics': {'messageTypeCount': {}, 'score': 0}, 'messages': []}
    expected = {'score': 0}
    assert analysis.parse_pylint(output_json) == expected


def test_parse_pylint_single_message(analysis):
    """Test the parse_pylint method of the Analysis class with a single message."""
    output_json = {
        'statistics': {'messageTypeCount': {}, 'score': 10},
        'messages': [{'messageId': 'C0103'}],
    }
    expected = {'score': 10, 'C0103': 1.0}
    assert analysis.parse_pylint(output_json) == expected


def test_parse_json_bandit_empty_output(analysis):
    """Test the parse_json_bandit method with empty output."""
    output_json = {'metrics': {'_totals': {}}, 'results': []}
    assert analysis.parse_json_bandit(output_json) == {}


def test_parse_json_bandit_single_issue(analysis):
    """Test the parse_json_bandit method with a single issue."""
    output_json = {'metrics': {'_totals': {'loc': 100}}, 'results': [{'test_id': 'B101'}]}
    assert analysis.parse_json_bandit(output_json) == {'loc': 100, 'B101': 1.0}


def test_parse_json_bandit_multiple_issues(analysis):
    """Test the parse_json_bandit method with multiple issues."""
    output_json = {
        'metrics': {'_totals': {'loc': 100}},
        'results': [{'test_id': 'B101'}, {'test_id': 'B102'}],
    }
    assert analysis.parse_json_bandit(output_json) == {'loc': 100, 'B101': 1.0, 'B102': 1.0}


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
