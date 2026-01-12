"""Tests for the GitRepo class in ml.sca.repo module."""

from pathlib import Path
from unittest.mock import Mock

from git import Repo
from github import Github
import pytest

from ml import Library
from ml.sca.repo import GitRepo

# pylint: disable=redefined-outer-name,protected-access,unused-argument


@pytest.fixture
def mock_library():
    """Create a mock Library object."""
    lib = Mock(spec=Library)
    lib.git_name = 'test-repo'
    lib.git_url = 'https://github.com/user/test-repo.git'
    lib.package_name = 'test'
    return lib


class TestGitRepoInit:
    """Tests for GitRepo initialization."""

    def test_init_with_github_url(self, mock_library: Mock, tmp_path: Path, mocker):
        """Test initialization with a GitHub URL."""

        mocker.patch('ml.sca.repo.Repo.clone_from')
        mocker.patch('ml.sca.repo.Github', return_value=Mock(spec=Github))
        mocker.patch('ml.sca.repo.GitRepo._fetch_github_stats')
        mocker.patch('ml.sca.repo.GitRepo._scrape_github_stats')
        mocker.patch('ml.sca.repo.GitRepo._time_since_update')

        git_repo = GitRepo(
            library=mock_library,
            clone_path=tmp_path,
            results_dir=tmp_path,
            github_token='fake-token',
        )
        assert git_repo.library == mock_library

    def test_init_with_non_github_url(self, mock_library: Mock, mocker):
        """Test initialization raises NotImplementedError for non-GitHub URLs."""
        mock_library.git_url = 'https://gitlab.com/user/test-repo.git'
        mocker.patch('ml.sca.repo.Github', return_value=Mock(spec=Github))

        with pytest.raises(NotImplementedError):
            GitRepo(library=mock_library, github_token='fake-token')

    def test_init_without_github_token_env_var(self, mock_library: Mock, mocker):
        """Test initialization raises EnvironmentError when token is missing."""
        mocker.patch.dict('os.environ', {}, clear=True)

        with pytest.raises(EnvironmentError):
            GitRepo(library=mock_library, github_token=None)


class TestGithubLogin:
    """Tests for GitHub authentication."""

    def test_github_login_with_token(self, mock_library: Mock, tmp_path: Path, mocker):
        """Test successful GitHub login with token."""
        mocker.patch('ml.sca.repo.Github', return_value=Mock(spec=Github))
        mocker.patch('ml.sca.repo.GitRepo.clone_and_scrape')
        mocker.patch('ml.sca.repo.Repo')

        git_repo = GitRepo(library=mock_library, clone_path=tmp_path, github_token='fake-token')
        result = git_repo._github_login('fake-token')  # Ensure this returns a valid object
        assert result is not None

    def test_github_login_with_env_variable(self, mock_library: Mock, tmp_path: Path, mocker):
        """Test GitHub login using environment variable."""
        mocker.patch('ml.sca.repo.Github', return_value=Mock(spec=Github))
        mocker.patch('ml.sca.repo.GitRepo.clone_and_scrape')
        mocker.patch('ml.sca.repo.Repo')
        mocker.patch.dict('os.environ', {'GITHUB_TOKEN': 'env-token'})

        git_repo = GitRepo(library=mock_library, clone_path=tmp_path, github_token=None)
        result = git_repo._github_login(None)
        assert result is not None

    def test_github_login_without_token(self, mock_library: Mock, mocker):
        """Test GitHub login raises error without token."""
        mocker.patch.dict('os.environ', {}, clear=True)

        with pytest.raises(EnvironmentError):
            mock_git_repo = Mock()
            GitRepo._github_login(mock_git_repo, None)


class TestCloneAndScrape:
    """Tests for cloning and scraping repositories."""

    def test_clone_new_repository(self, mock_library: Mock, tmp_path: Path, mocker):
        """Test cloning a new repository."""
        mocker.patch('ml.sca.repo.Repo')
        mocker.patch('ml.sca.repo.Github', return_value=Mock(spec=Github))
        mocker.patch('ml.sca.repo.GitRepo._fetch_github_stats')
        mocker.patch('ml.sca.repo.GitRepo._scrape_github_stats')
        mocker.patch('ml.sca.repo.GitRepo._time_since_update')

        git_repo = GitRepo(library=mock_library, clone_path=tmp_path, github_token='fake-token')
        assert git_repo.repo is not None

    def test_use_existing_repository(self, mock_library: Mock, tmp_path: Path, mocker):
        """Test using an existing repository."""
        mocker.patch('ml.sca.repo.Repo')
        mocker.patch('ml.sca.repo.Github', return_value=Mock(spec=Github))
        mocker.patch('ml.sca.repo.GitRepo._fetch_github_stats')
        mocker.patch('ml.sca.repo.GitRepo._scrape_github_stats')
        mocker.patch('ml.sca.repo.GitRepo._time_since_update')

        tmp_path.mkdir(exist_ok=True)
        (tmp_path / ".git").mkdir(exist_ok=True)

        git_repo = GitRepo(library=mock_library, clone_path=tmp_path, github_token='fake-token')
        assert git_repo.repo is not None


class TestFromPackageName:
    """Tests for from_package_name class method."""

    def test_from_package_name_valid(self, mock_library: Mock, mocker):
        """Test creating GitRepo from valid package name."""
        mocker.patch('ml.sca.repo.all_libraries').__getitem__.return_value = mock_library
        mocker.patch('ml.sca.repo.Github', return_value=Mock(spec=Github))
        mocker.patch('ml.sca.repo.GitRepo._fetch_github_stats')
        mocker.patch('ml.sca.repo.Repo')
        with pytest.raises(ValueError):
            GitRepo.from_package_name('test-lib', github_token='fake-token')

    def test_from_package_name_invalid(self, mocker):
        """Test creating GitRepo from invalid package name."""
        mocker.patch('ml.sca.repo.all_libraries').__contains__.return_value = False
        with pytest.raises(ValueError):
            GitRepo.from_package_name('nonexistent-lib', github_token="fake-token")


class TestFetchBasicInfo:
    """Tests for fetching basic repository information."""

    def test_fetch_basic_info(self, mock_library: Mock, tmp_path: Path, mocker):
        """Test fetching basic repository information."""
        mock_repo = Mock(spec=Repo)
        mock_repo.active_branch = 'main'
        mock_repo.iter_commits.return_value = [Mock(), Mock(), Mock()]
        mock_repo.num_commits = 3
        mock_repo.head.commit.committed_datetime.strftime.return_value = '2023-01-01 00:00:00'

        mocker.patch('ml.sca.repo.Github', return_value=Mock(spec=Github))
        mocker.patch('ml.sca.repo.GitRepo._fetch_github_stats')
        mocker.patch('ml.sca.repo.GitRepo._scrape_github_stats')
        mocker.patch('ml.sca.repo.GitRepo._time_since_update')
        mocker.patch('ml.sca.repo.Repo', return_value=mock_repo)
        git_repo = GitRepo(library=mock_library, clone_path=tmp_path, github_token='fake-token')

        assert hasattr(git_repo, 'active_branch')
        assert hasattr(git_repo, 'lines_of_code')
        assert isinstance(git_repo.num_commits, int)
        assert isinstance(git_repo.num_contributors, int)
        assert hasattr(git_repo, 'updated_at')
