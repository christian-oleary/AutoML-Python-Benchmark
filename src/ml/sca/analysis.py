"""Code for performing static code analysis (SCA) on a directory."""

import json
import os
import subprocess
from pathlib import Path

from git import Repo
from git.exc import InvalidGitRepositoryError

import pandas as pd

from ml.logs import logger
from ml.util import Utils


IGNORED_SONAR_METRICS = ['quality_profiles', 'quality_gate_details', 'alert_status']


class GitRepo:
    """Class to represent a Git repository."""

    name: str
    path: str | Path
    url: str
    commit_count: int
    contributors: list[str]
    latest_commit: str
    lines_of_code: int

    def __init__(self, repo_path: str | Path):
        """Initialize the Git repository.

        :param str | Path repo_path: Path to repository.
        """
        self.path = repo_path
        self.repo = Repo(self.path)

        self.commit_count = self.get_commit_count()
        self.contributors = self.get_contributors()
        self.lines_of_code = self.get_lines_of_code()
        self.name = self.get_repo_name()
        self.num_contributors = self.count_contributors()
        self.url = self.get_repo_url()

    def get_branches(self) -> list[str]:
        """Get the branches in the repository.

        :return list[str]: The branches in the repository.
        """
        return [str(branch) for branch in self.repo.branches]

    def get_commit_count(self, **kwargs) -> int:
        """Get the number of commits in the repository.

        :return int: The number of commits in the repository.
        """
        self.commit_count = len(list(self.repo.iter_commits(**kwargs)))
        return self.commit_count

    def get_contributors(self) -> list[str]:
        """Get the contributors to the repository.

        :return list[str]: The contributors to the repository.
        """
        contributors = set()
        for commit in self.repo.iter_commits():
            contributors.add(str(commit.author.email))
        self.contributors = list(contributors)
        return self.contributors

    def get_latest_commit(self) -> str:
        """Get the latest commit in the repository."""
        self.latest_commit = str(self.repo.head.commit)
        return self.latest_commit

    def get_repo_name(self) -> str:
        """Get the name of the repository.

        :return str: The name of the repository.
        """
        self.repo_name = self.repo.remotes.origin.url.split('/')[-1].replace('.git', '')
        return self.repo_name

    def get_lines_of_code(self) -> int:
        """Get the number of lines of code in the repository.

        :return int: The number of lines of code in the repository
        """
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

    def get_repo_url(self) -> str:
        """Get the URL of the repository.

        :return str: The URL of the repository.
        """
        self.repo_url = self.repo.remotes.origin.url
        return self.repo_url

    def count_contributions(self, email: str) -> int:
        """Count the number of contributions by a given email address.

        :param str email: The email address to count contributions for.
        :return int: The number of contributions by the email address.
        """
        count = 0
        for commit in self.repo.iter_commits():
            if commit.author.email == email:
                count += 1
        return count

    def count_contributors(self) -> int:
        """Count the number of contributors to the repository.

        :return int: The number of contributors to the repository.
        """
        return len(self.get_contributors())


class Analysis:
    """Class for performing static code analysis (SCA) on a directory."""

    def __init__(self, input_dir: str | Path, output_dir: str | Path | None = None):
        """Analysis class for static code analysis (SCA) on a directory.

        :param str | Path directory: Directory assumed to represent cloned repository or contain cloned repositories.
        :param str | Path output_dir: Directory to save the results of the analysis, defaults to None.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = output_dir

    def run(self) -> list[dict]:
        """Run the static code analysis."""
        self.start_time = pd.Timestamp.now().isoformat()

        # Check if the directory is a Git repository
        if self.is_git_repo(self.input_dir):
            self.results = [self.analyze_repo(self.input_dir)]

        # Otherwise, assume the directory contains Git repositories
        else:
            repos = [d for d in self.input_dir.iterdir() if self.is_git_repo(d)]
            if len(repos) == 0:
                raise ValueError(f'No Git repositories found in directory: {self.input_dir}')
            self.results = [self.analyze_repo(repo) for repo in repos]

        # Save the results to a file if an output directory is provided
        self.end_time = pd.Timestamp.now().isoformat()
        if self.output_dir:
            logger.debug(f'Saving results to {self.output_dir}...')
            self.save_results(self.output_dir)
        return self.results

    def analyze_repo(self, target_dir: str | Path) -> dict:
        """Analyze a single Git repository in the specified directory.

        :param str | Path target_dir: Path to the Git repository.
        :return dict: The analysis of the Git repository.
        """
        logger.info(f'Analyzing {target_dir}...')
        repo = GitRepo(target_dir)

        # Run analysis on the repository
        sonar_results = self.parse_sonar_scanner_json(repo, self.output_dir)
        results = {
            'name': repo.name,
            'path': repo.path,
            **{f'git__{k}': v for k, v in self.git_analysis(repo).items()},
            **{f'sonar__{k}': v for k, v in sonar_results.items()},
            **{f'pylint__{k}': v for k, v in self.run_pylint(repo).items()},
        }
        return results

    def run_pylint(self, repo: GitRepo) -> dict:
        """Run pylint on the specified directory.

        :param GitRepo repo: The Git repository object.
        :return dict: The frequencies of pylint errors.
        """
        logger.debug(f'Running pylint on {repo.name}...')

        # Temporary pylint output file. Deleted later.
        pylint_file: str | Path = f'pylint_{repo.name}.json'

        # File to save formatted results
        if self.output_dir:
            results_file = Path(self.output_dir, 'pylint', pylint_file)
            # Load the pylint output from a file if possible
            if results_file and results_file.exists():
                logger.debug(f'Output file already exists: {results_file}')
                frequencies = json.loads(results_file.read_text(encoding='utf-8'))
                return frequencies
        else:
            results_file = None

        # Run pylint command
        command = ['pylint', '-ry', '--output-format=json2', f'--output={pylint_file}', '.']
        result = subprocess.run(command, capture_output=True, check=False, cwd=repo.path, text=True)
        pylint_file = Path(repo.path, pylint_file)
        logger.debug(result)

        # Load the pylint output from the temporary file
        with open(pylint_file, 'r', encoding='utf-8') as f:
            pylint_output = json.load(f)

        # Count message frequencies
        frequencies = {}
        for m in pylint_output['messages']:
            message_id = m['messageId']
            if message_id not in frequencies:
                frequencies[message_id] = 0
            frequencies[message_id] += 1

        # Save results to a file if an output directory is provided
        if self.output_dir and results_file:
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(pylint_output, f, indent=4)
            logger.info(f'Pylint output saved to {results_file}')

        os.remove(pylint_file)
        return frequencies

    def git_analysis(self, repo: GitRepo, verbose: bool = False) -> dict:
        """Analyze the Git repository.

        :param GitRepo repo: The Git repository object.
        :param bool verbose: Whether to include additional information, defaults to False.
        :return dict: The analysis of the Git repository.
        """
        logger.debug(f'Git analysis of {repo.path}...')
        analysis = {
            'repository': repo.name,
            'path': repo.path,
            'lines_of_code': repo.lines_of_code,
            'num_commits': repo.commit_count,
            'num_contributors': repo.num_contributors,
        }
        if verbose:
            analysis = {
                **analysis,
                'branches': ', '.join(repo.get_branches()),
                'contributors': ', '.join(repo.get_contributors()),
                'latest_commit': repo.get_latest_commit(),
            }
        return analysis

    def parse_sonar_scanner_json(self, repo: GitRepo, output_dir: str | Path | None):
        """Parse the JSON output from SonarScanner.

        :param GitRepo repo: The Git repository object.
        :param str | Path | None output_dir: Output directory, defaults to None.
        :return dict: The SonarScanner analysis results.
        """
        if output_dir is None:  # No sonar scanner files specified
            return {}

        # Assume the sonar scanner output is: output_dir/sonar/[repo_name]/
        sonar_dir = Path(output_dir, 'sonar', repo.get_repo_name())
        if not sonar_dir.is_dir():
            logger.debug(f'No SonarScanner output found for {repo.name}')
            return {}

        # Find the measures file in the SonarScanner output
        measures_file = Path(sonar_dir, 'measures.json')
        if not measures_file.is_file():
            logger.debug(f'No measures file found for {repo.name}')
            return {}

        # Load and return parsed measures if available
        parsed_measures_file = Path(output_dir, 'sonar_parsed', f'sonar_{repo.name}.json')
        if parsed_measures_file.exists():
            logger.debug(f'Loading parsed measures from {parsed_measures_file}')
            return json.loads(parsed_measures_file.read_text(encoding='utf-8'))

        # Read measures (lines of JSON) from SonarScanner output file
        logger.debug(f'Parsing SonarScanner output: {measures_file}')
        results: dict[str, float | int] = {}
        with open(measures_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Iterate through the lines of measures
        for line in lines:
            # Parse the JSON line and extract the measures
            measures = json.loads(line)['component']['measures']

            # Filter out ignored metrics
            measures = [m for m in measures if m['metric'] not in IGNORED_SONAR_METRICS]

            # Record the measures as floats
            for measure in measures:
                # Handle ncloc_language_distribution
                if measure['metric'] == 'ncloc_language_distribution':
                    measure['value'] = self.calculate_python_percentage(measure['value'])

                # Record last_commit_date as int
                if measure['metric'] == 'last_commit_date':
                    results[measure['metric']] = int(measure['value'])
                else:
                    # Record the measure as a float
                    results[measure['metric']] = float(measure['value'])

        # Save the parsed measures to a file
        parsed_measures_file.parent.mkdir(parents=True, exist_ok=True)
        with open(parsed_measures_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        return results

    def calculate_python_percentage(self, language_distribution: str):
        """Calculate the percentage of Python code in the repository.

        :param str language_distribution: The language distribution string from SonarScanner.
        :return float: The percentage of Python code in the repository.
        """
        # Split by language
        counts = language_distribution.split(';')
        # Create a dictionary of counts
        counts_dict = {c.split('=')[0]: int(c.split('=')[1]) for c in counts}
        # Calculate the percentage of Python code
        python_percentage = counts_dict.get('py', 0) / sum(counts_dict.values())
        return python_percentage

    def is_git_repo(self, path) -> bool:
        """Check if a given path is a valid Git repository.

        :param path: The path to check.
        :return bool: True if the path is a valid Git repository, False otherwise.
        """
        # Check if the path is a directory
        if path is None or not Path(path).is_dir():
            return False
        # Check if the directory is a Git repository
        try:
            Repo(path)
            return True
        except InvalidGitRepositoryError:
            return False

    def save_results(self, output_dir: str | Path) -> pd.DataFrame:
        """Save the results of the analysis to a file.

        :param str output_dir: Path to save the analysis
        :return pd.DataFrame: The results as a pandas DataFrame.
        """
        if self.output_dir is None:
            raise ValueError('No output directory specified')

        # Save results to a CSV file
        results_file = Path(output_dir, 'results.csv')
        df = pd.DataFrame(self.results)
        df.to_csv(results_file, index=False)
        logger.info(f'Saved results to {results_file}')

        # Save metadata to a JSON file
        metadata = {
            'input_dir': str(self.input_dir),
            'num_repos': len(self.results),
            'start_time': str(self.start_time),
            'end_time': str(self.end_time),
            'duration': str(pd.Timestamp(self.end_time) - pd.Timestamp(self.start_time)),
        }
        metadata_file = Path(output_dir, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f'Saved metadata to {metadata_file}: {metadata}')

        # Save correlation matrix to CSV file
        if len(df) > 1:
            path = str(Path(self.output_dir, 'correlation_heatmap.csv'))
            Utils.save_heatmap(df, path, path.replace('.csv', '.png'), columns='all')
        return df
