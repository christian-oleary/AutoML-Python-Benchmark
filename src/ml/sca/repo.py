"""Code for scraping git repository information (stars, forks, issues, etc.)."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Optional

from git import Repo
from git.exc import InvalidGitRepositoryError
from github import Auth, Github
from loguru import logger
import pandas as pd
import requests

from ml import DEFAULT_REPOSITORIES_PATH, Library, all_libraries


class GitRepo:
    """Class to represent a Git repository."""

    library: Library
    path: str | Path
    days_since_update: int

    df: Optional[pd.DataFrame] = None
    g: Optional[Github] = None
    github_token: Optional[str] = None
    repo: Optional[Repo] = None

    # From GitPython
    active_branch: str
    branches: list[str]
    lines_of_code: int
    num_branches: int
    num_commits: int
    num_contributors: int
    num_files: int
    num_files_python: int
    updated_at: str

    # From PyGithub
    created_at: str
    issues_closed: int
    issues_open: int
    issues_total: int
    license: str
    num_forks: int
    num_releases: int
    num_stars: int
    num_tags: int
    num_watchers: int

    def __init__(
        self,
        library: Library,
        clone_path: Optional[str | Path] = None,
        github_token: Optional[str] = None,
        results_dir: Optional[str | Path] = None,
        skip_existing: bool = False,
        **kwargs,
    ):
        """Initialize the GitRepo object. Clone and scrape the repository.

        :param Library library: Library object representing the Git repository.
        :param str | Path | None clone_path: Local path to clone the repository to.
        :param str | None github_token: GitHub token for authentication.
        :param str | Path | None results_dir: Directory to save results to.
        :param bool skip_existing: Skip analysis if results already exist.
        """
        self.library = library
        self.github_token = github_token
        if results_dir is not None:
            results_dir = Path(results_dir)
            self.csv_path = results_dir / f'{self.library.package_name}.csv'

        # Set the local OS path to clone the repository to
        if clone_path is None:
            clone_path = Path(DEFAULT_REPOSITORIES_PATH, str(library.git_name))
        self.path = Path(clone_path)

        # Authenticate to GitHub if needed
        if 'github' in self.library.git_url.lower():
            self.g = self._github_login(github_token)
        else:
            raise NotImplementedError('Only GitHub repositories are supported currently.')

        # Clone and scrape the repository
        self.clone_and_scrape(
            github_token=github_token,
            results_dir=results_dir,
            skip_existing=skip_existing,
            **kwargs,
        )

    def _github_login(self, github_token: str | None) -> Optional[Github]:
        """Login to GitHub using a token.

        :param str | None github_token: GitHub token.
        :return Github | None: Authenticated GitHub object or None if authentication failed.
        """
        # Attempt to get from environment variable if missing
        if github_token is None:
            github_token = os.getenv('GITHUB_TOKEN')
        # Check if token is provided
        if github_token is None:
            raise EnvironmentError('GITHUB_TOKEN environment variable not set.')
        # Authenticate to GitHub using the token
        g = Github(auth=Auth.Token(github_token))
        return g

    def clone_and_scrape(self, **kwargs) -> GitRepo:
        """Clone the repository and scrape its information.

        :return GitRepo: The GitRepo object with scraped information.
        """
        github_token = kwargs.get('github_token', None)
        results_dir = kwargs.get('results_dir', None)
        skip_existing = kwargs.get('skip_existing', False)

        # Specify path to save results
        if results_dir is not None:
            results_dir = Path(results_dir)
            csv_path = results_dir / f'{self.library.package_name}.csv'

            # Check if results already exist and skip if needed
            if skip_existing and csv_path.exists():
                logger.debug(f'Results already exist at {csv_path}. Skipping...')
                self.df = pd.read_csv(csv_path)
                return self

        if not isinstance(self.path, Path):
            raise ValueError('Target path for cloning the repository is not set.')

        # Clone the repository if it does not exist
        if not self.path.exists() or not os.listdir(self.path):
            logger.debug(f'Cloning from {self.library.git_url} to: {self.path}...')
            self.repo = Repo.clone_from(self.library.git_url, self.path)
        else:
            logger.debug(f'Repository already exists at: {self.path}')
            self.repo = Repo(self.path)

        # Pull the latest changes
        logger.debug(f'Pulling latest changes for {self.library.git_name}...')
        self.repo.remotes.origin.pull()

        # Verify that the repository is valid
        try:
            _ = self.repo.git_dir
        except InvalidGitRepositoryError as e:
            logger.error(f'Invalid Git repository at {self.path}: {e}')
            raise

        # Get basic repository information using GitPython
        self._fetch_basic_info()

        # Fetch GitHub statistics using PyGithub
        self._fetch_github_stats(github_token)
        if not hasattr(self, 'issues_closed') or self.issues_closed in (None, 0):
            logger.debug('Missing closed issues. Scraping from web page...')
            self._scrape_github_stats(github_token)

        # Save to CSV if results directory provided
        if results_dir is not None:
            self.to_csv(csv_path)
        return self

    def to_csv(self, csv_path: str | Path) -> None:
        """Save the repository information to a CSV file.

        :param str | Path csv_path: Path to save the CSV file.
        """
        self.df = self._as_dataframe()

        # Create results directory
        results_dir = Path(csv_path).parent
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save the DataFrame to a CSV file
        csv_path = results_dir / f'{self.library.package_name}.csv'
        self.df.to_csv(csv_path, index=False)
        logger.debug(f'Saved to {csv_path}')

    @classmethod
    def from_package_name(cls, name: str, **kwargs) -> GitRepo:
        """Scrape the repository for a given library.

        :param str name: Name of library to scrape.
        """
        logger.info(f'Creating GitRepo for library: {name}')
        if name not in all_libraries:
            raise ValueError(f'Library {name} not found in known libraries.')
        return cls(library=all_libraries[name], **kwargs)

    def _fetch_basic_info(self) -> None:
        """Fetch basic repository information using GitPython."""
        if self.repo is None:
            raise ValueError('Git repository is not initialized.')
        # Branches
        self.active_branch = str(self.repo.active_branch)  # Overwritten by GitHub if available
        # self.num_branches = len([str(branch) for branch in self.repo.branches])  # Returns 1
        # Commits
        self.num_commits = len(list(self.repo.iter_commits()))
        self._get_latest_commit()  # Overwritten by GitHub data if available
        self._time_since_update()
        self._get_num_contributors()
        self._get_lines_of_code()
        self._get_num_python_files()

    def _get_num_contributors(self) -> int:
        """Get the contributors to the repository.

        :return int: The number of unique contributors to the repository.
        """
        if self.repo is None:
            raise ValueError('Git repository is not initialized.')

        contributors = set()
        for commit in self.repo.iter_commits():
            contributors.add(str(commit.author.email))
        self.num_contributors = len(contributors)
        return self.num_contributors

    def _get_latest_commit(self) -> str:
        """Get the latest commit in the repository."""
        if self.repo is None:
            raise ValueError('Git repository is not initialized.')
        timestamp = self.repo.head.commit.committed_datetime
        self.updated_at = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        return self.updated_at

    def _time_since_update(self) -> int:
        """Get the time since the repository was last updated in days.

        :return int: The time since the repository was last updated in days.
        """
        if not hasattr(self, 'updated_at') or self.updated_at is None:
            self.updated_at = self._get_latest_commit()

        updated_at_dt = datetime.strptime(self.updated_at, '%Y-%m-%d %H:%M:%S')
        delta = datetime.now() - updated_at_dt
        self.days_since_update = delta.days
        return self.days_since_update

    def _get_lines_of_code(self) -> int:
        """Get the number of lines of code in the repository.

        :return int: The number of lines of code in the repository
        """
        if self.repo is None:
            raise ValueError('Git repository is not initialized.')

        self.lines_of_code = 0
        # Iterate through the files in the repository
        for root, _, files in os.walk(self.repo.working_dir):
            # Only count Python files
            for file in [f for f in files if f.endswith('.py')]:
                # Count the lines of code in the file
                with open(Path(root, file), 'r', encoding='utf-8') as f:
                    # Try to read file
                    try:
                        num_lines = len(f.readlines())
                    except UnicodeDecodeError:
                        logger.debug(f'Could not read {file}. Skipping...')
                        continue
                    # Count lines of code
                    self.lines_of_code += num_lines
        return self.lines_of_code

    def _get_num_python_files(self):
        """Get the number of Python files in the repository."""
        if self.repo is None:
            raise ValueError('Git repository is not initialized.')
        self.num_files = 0
        self.num_files_python = 0
        # Iterate through the files in the repository
        for _, __, files in os.walk(self.repo.working_dir):
            self.num_files += len(files)
            self.num_files_python += len([f for f in files if f.endswith('.py')])

    def _fetch_github_stats(self, github_token: Optional[str] = None) -> None:
        """Fetch GitHub statistics for the repository.

        :param str | None github_token: GitHub token for authentication.
        """
        # Authenticate to GitHub
        g = self._github_login(github_token)
        if g is None:
            return
        # Get the repository from GitHub
        github_repo = g.get_repo(self.library.git_url.split('github.com/')[-1].replace('.git', ''))

        # Creation and update dates
        self.created_at = github_repo.created_at.strftime('%Y-%m-%d %H:%M:%S')
        self.updated_at = github_repo.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        # Branches
        self.active_branch = github_repo.default_branch
        self.num_branches = github_repo.get_branches().totalCount
        # Stars, Forks, Watchers and Clones
        self.num_forks = github_repo.forks_count
        self.num_stars = github_repo.stargazers_count
        self.num_watchers = github_repo.subscribers_count
        # Issues
        self.issues_closed = github_repo.get_issues(state='closed').totalCount  # NOTE May return 0
        self.issues_open = github_repo.open_issues_count
        self.issues_total = github_repo.get_issues().totalCount  # NOTE May return 0
        # License, Releases and Tags
        self.license = github_repo.get_license().license.spdx_id
        self.num_releases = github_repo.get_releases().totalCount
        self.num_tags = github_repo.get_tags().totalCount

        g.close()  # Close connections after use

    def _scrape_github_stats(self, github_token: Optional[str] = None) -> None:
        """Scrape GitHub statistics for the repository as a fallback."""
        headers = {
            'Accept': 'application/vnd.github+json',
            'Authorization': f'Bearer {github_token}',
        }

        def count_issues():
            """Count open and closed issues using GitHub API."""
            counts = {'open': 0, 'closed': 0}

            namespace = self.library.git_url.split('github.com/')[-1].replace('.git', '')
            url = f'https://api.github.com/repos/{namespace}/issues'
            params = {'state': 'all', 'per_page': 100}

            while url:
                # Make the API request
                r = requests.get(url, headers=headers, params=params, timeout=30)
                r.raise_for_status()
                # Count issues
                for item in r.json():
                    if 'pull_request' not in item:  # exclude PRs
                        counts[item['state']] += 1
                # Get next page from Link header
                url = None
                link = r.headers.get('Link')
                if link:
                    for part in link.split(','):
                        if 'rel="next"' in part:
                            url = part.split(';')[0].strip('<> ')
                sleep(1)  # To respect rate limits
            return counts

        counts = count_issues()
        self.issues_open = counts['open']
        self.issues_closed = counts['closed']
        self.issues_total = counts['open'] + counts['closed']

    def _as_dataframe(self) -> pd.DataFrame:
        """Save the repository information to a CSV file.

        :return pd.DataFrame: DataFrame containing the repository information.
        """

        # Create a DataFrame from the repository information
        def _filter(key: str) -> bool:
            filters = (
                not key.startswith('_'),
                not callable(getattr(self, key)),
                key not in ('df', 'g', 'github_token', 'repo'),
            )
            return all(filters)

        data = {k: [v] for k, v in self.__dict__.items() if _filter(k)}
        self.df = pd.DataFrame(data)
        return self.df

    def __repr__(self) -> str:
        return f'GitRepo({self.library.package_name})'

    def __str__(self) -> str:
        text = f'GitRepo: {self.library.git_url}:'
        properties = [
            p for p in dir(self) if not p.startswith('_') and not callable(getattr(self, p))
        ]
        properties = [p for p in properties if p not in ('df', 'g', 'github_token', 'repo')]
        # Get the longest name length for formatting
        longest_name_length = max(len(name) for name in properties)
        # Add each property to the string
        for name in properties:
            value = getattr(self, name)
            text += f'\n - {name.ljust(longest_name_length)} -> {value}'
        text += '\nNotes:\n'
        text += ' * lines_of_code counts only .py files.\n'
        text += ' * num_contributors counts unique email addresses from commit history.\n'
        return text


if __name__ == '__main__':
    # Example usage:

    CLONE_PATH = Path('repositories/')
    RESULTS_DIR = Path('results/sca/git_analysis/')
    SKIP_EXISTING = True
    TOKEN = os.getenv('GITHUB_TOKEN')
    if TOKEN is None:
        raise EnvironmentError('GITHUB_TOKEN environment variable not set.')

    dataframes = []
    # libraries = ['autosklearn']
    libraries = list(all_libraries.keys())

    for library_name in libraries:
        repo = GitRepo.from_package_name(
            name=library_name,
            github_token=TOKEN,
            clone_path=CLONE_PATH,
            results_dir=RESULTS_DIR,
            skip_existing=SKIP_EXISTING,
        )
        dataframes.append(repo.df)
        print(repo)

    # Join dataframes
    if len(dataframes) > 0:
        df_all = pd.concat(dataframes, ignore_index=True)
        df_all.to_csv(RESULTS_DIR / '1_ALL_LIBRARIES.csv', index=False)
