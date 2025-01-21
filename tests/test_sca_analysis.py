"""Tests for the sca.analysis module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from git import InvalidGitRepositoryError
import pandas as pd
import pytest

from ml.sca.analysis import Analysis, GitRepo

# pylint: disable=redefined-outer-name,unused-argument

# DEFAULTS
DEFAULT_BRANCHES = ['main']
DEFAULT_COMMIT = 'latest_commit'
DEFAULT_EMAIL = 'test@example.com'
DEFAULT_NAME = 'repo'
DEFAULT_PATH = 'path/to/repo'
DEFAULT_URL = 'https://github.com/user/repo.git'

DEFAULT_PATH_TO_INPUT = 'path/to/input'
DEFAULT_PATH_TO_OUTPUT = 'path/to/output'


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
def git_repo(mock_repo):
    """Fixture to return a GitRepo object."""
    return GitRepo(repo_path=DEFAULT_PATH)


def test_git_repo_initialization(git_repo):
    """Test the initialization of a GitRepo object."""
    assert git_repo.path == DEFAULT_PATH
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
def analysis():
    """Fixture to return an Analysis object."""
    return Analysis(input_dir=DEFAULT_PATH_TO_INPUT, output_dir=DEFAULT_PATH_TO_OUTPUT)


def test_analysis_initialization(analysis):
    """Test the initialization of an Analysis object."""
    assert str(Path(analysis.input_dir)) == str(Path(DEFAULT_PATH_TO_INPUT))
    assert str(Path(analysis.output_dir)) == str(Path(DEFAULT_PATH_TO_OUTPUT))


def test_analysis_run(analysis, mock_repo):
    """Test the run method of the Analysis class."""
    with (
        patch.object(analysis, 'is_git_repo', return_value=True),
        patch.object(analysis, 'analyze_repo', return_value={'name': DEFAULT_NAME}),
        patch.object(analysis, 'save_results'),
    ):
        results = analysis.run()
        assert results == [{'name': DEFAULT_NAME}]


def test_analysis_analyze_repo(analysis, mock_repo):
    """Test the analyze_repo method of the Analysis class."""
    with (
        patch.object(analysis, 'parse_sonar_scanner_json', return_value={}),
        patch.object(analysis, 'git_analysis', return_value={}),
        patch.object(analysis, 'run_pylint', return_value={}),
    ):
        results = analysis.analyze_repo(DEFAULT_PATH)
        assert results['name'] == DEFAULT_NAME
        assert results['path'] == DEFAULT_PATH
        # assert results['git__repository'] == DEFAULT_NAME
        # assert results['git__path'] == DEFAULT_PATH
        # assert results['git__lines_of_code'] == 0
        # assert results['git__num_commits'] == 1
        # assert results['git__num_contributors'] == 1
        # assert results['sonar__'] == {}
        # assert results['pylint__'] == {}


def test_analysis_run_pylint(analysis, git_repo):
    """Test the run_pylint method of the Analysis class."""
    with (
        patch('subprocess.run') as mock_run,
        patch('builtins.open', new_callable=MagicMock),
        patch('json.load', return_value={'messages': []}),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        results = analysis.run_pylint(git_repo)
        assert results == {}


def test_analysis_git_analysis(analysis, git_repo):
    """Test the git_analysis method of the Analysis class."""
    results = analysis.git_analysis(git_repo)
    assert results == {
        'repository': DEFAULT_NAME,
        'path': DEFAULT_PATH,
        'lines_of_code': 0,
        'num_commits': 1,
        'num_contributors': 1,
    }


def test_analysis_parse_sonar_scanner_json(analysis, git_repo):
    """Test the parse_sonar_scanner_json method of the Analysis class."""
    with (
        patch('builtins.open', new_callable=MagicMock),
        patch('json.load', return_value={'component': {'measures': []}}),
    ):
        results = analysis.parse_sonar_scanner_json(git_repo, DEFAULT_PATH_TO_OUTPUT)
        assert results == {}


def test_analysis_is_git_repo(analysis):
    """Test the is_git_repo method of the Analysis class."""
    with patch('ml.sca.analysis.Repo') as mock_repo:
        mock_repo.side_effect = InvalidGitRepositoryError
        assert not analysis.is_git_repo(DEFAULT_PATH)


def test_analysis_save_results(analysis):
    """Test the save_results method of the Analysis class."""
    with patch('pandas.DataFrame.to_csv'), patch('json.dump'), patch('ml.util.Utils.save_heatmap'):
        analysis.results = [{'name': DEFAULT_NAME}]
        analysis.start_time = 0
        analysis.end_time = 1
        df = analysis.save_results(DEFAULT_PATH_TO_OUTPUT)
        assert isinstance(df, pd.DataFrame)
