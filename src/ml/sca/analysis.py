"""Code for performing static code analysis (SCA) on a directory."""

import json
import os
import subprocess  # nosec
from pathlib import Path
from typing import Any

from defusedxml import ElementTree
from git import Repo
from git.exc import InvalidGitRepositoryError

import pandas as pd

from ml.logs import logger

COVERAGE_MEANS = ['line-rate', 'branch-rate', 'complexity']
COVERAGE_SUMS = ['lines-covered', 'lines-valid', 'branches-covered', 'branches-valid']
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

    def __init__(self, input_dir: str | Path, output_dir: str | Path):
        """Analysis class for static code analysis (SCA) on a directory.

        :param str | Path input_dir: Directory of cloned repository or containing cloned repositories.
        :param str | Path output_dir: Directory to save the results of the analysis.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir, 'sca')
        self.results: list[dict] = []
        self.start_time: str = ''
        self.end_time: str = ''

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

    def analyze_repo(
        self,
        target_dir: str | Path,
        skip_existing_sonar: bool = False,
        skip_existing_sca: bool = True,
    ) -> dict:
        """Analyze a single Git repository in the specified directory.

        :param str | Path target_dir: Path to the Git repository.
        :param bool skip_existing_sonar: Whether to skip existing Sonar-Scanner results, defaults to False.
        :param bool skip_existing_sca: Whether to skip existing CLI SCA tool results, defaults to True.
        :return dict: The analysis of the Git repository.
        """
        logger.info(f'Analyzing {target_dir}...')
        repo = GitRepo(target_dir)

        # Run analysis on the repository
        sonar_results = self.parse_sonar_scanner_json(repo, self.output_dir, skip_existing_sonar)
        results = {
            'name': repo.name,
            'path': repo.path,
            **{f'coverage__{k}': v for k, v in self.read_coverage_xml(repo).items()},
            **{f'git__{k}': v for k, v in self.git_analysis(repo, verbose=False).items()},
            **{f'sonar__{k}': v for k, v in sonar_results.items()},
        }

        # Run CLI tools on the repository
        for tool, command in self.build_commands(repo.name).items():
            outputs = self.run_cli_command(tool, command, repo, skip_existing_sca)
            results.update({f'{tool}__{k}': v for k, v in outputs.items()})
            logger.info(f'{tool} analysis complete for {repo.name}')
        return results

    def build_commands(self, repo_name: str | Path) -> dict:
        """Build the CLI commands for the static code analysis tools.

        :param str | Path repo_name: The name of the repository.
        :return dict: The CLI commands for the static code analysis tools.
        """
        filename = f'_{repo_name}.json'
        # fmt: off
        commands = {
            'bandit': ['bandit', '-r', '--format', 'json', '-o', f'bandit{filename}', '.'],
            # 'flake8': ['flake8', '--format=json-pretty', f'--output-file=flake8{filename}', '.'],  # JSON errors
            'prospector': [
                'prospector', '-o', f'json:prospector{filename}',
                '--tool', 'dodgy', '--tool', 'mypy', '--tool', 'profile-validator',
                '--tool', 'pyright', '--tool', 'pyroma', '--tool', 'vulture',
                '.'
            ],
            'pylint': ['pylint', '-ry', '--output-format=json2', f'--output=pylint{filename}', '.'],
            'radon-cc': ['radon', 'cc', '-ja', '-O', f'radon-cc{filename}', '.'],
            'radon-hal': ['radon', 'hal', '-j', '-O', f'radon-hal{filename}', '.'],
            'radon-mi': ['radon', 'mi', '-j', '-O', f'radon-mi{filename}', '.'],
            'radon-raw': ['radon', 'raw', '-j', '-O', f'radon-raw{filename}', '.'],
            'ruff': ['ruff', 'check', '--statistics', '--output-file', f'ruff{filename}', '--output-format', 'json'],
        }
        # fmt: on
        return commands

    def run_cli_command(
        self,
        tool: str,
        command: list,
        repo: GitRepo,
        skip_existing: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Run a CLI command on the specified directory.

        :param str tool: The tool to run the CLI command for.
        :param list command: The CLI command to run.
        :param GitRepo repo: The Git repository object.
        :param bool skip_existing: Whether to skip existing results, defaults to True.
        :param bool verbose: Whether to include additional information, defaults to False.
        :raises ValueError: If the tool is not supported.
        :return dict: The frequencies of the tool errors.
        """
        # Check if the results file already exists
        logger.info(f'{tool} - {repo.name}')
        result_path, frequencies, json_file = self.check_results_file(tool, repo, skip_existing)

        # Return the frequencies if they already exist
        if frequencies and skip_existing:
            return frequencies

        # Run the CLI command
        logger.info(f'Running {tool} on {repo.name} using command: {command}')
        result = subprocess.run(  # nosec
            command,
            capture_output=True,
            check=False,
            cwd=repo.path,
            text=True,
        )
        if verbose:
            logger.debug(f'{tool} -> stdout: \n{result.stdout}')
            if result.returncode != 0:
                logger.error(f'{tool} -> stderr: \n{result.stderr}')
        else:
            logger.debug(f'{tool}: \n{result}')

        # Ensure output file was created
        if not json_file.is_file():
            raise ValueError(f'Output file not found: {json_file}')

        # Parse the results file
        frequencies = self.parse_results_file(tool, result_path, json_file)
        return frequencies

    def read_coverage_xml(self, repo: GitRepo):
        """Read the coverage XML file from the repository if present.

        :param GitRepo repo: The Git repository object.
        :param str pattern: The pattern to search for, defaults to 'coverage.xml'.
        """
        logger.info(f'Reading coverage XML file for {repo.name}...')
        coverage_results: dict[str, Any] = {}
        # Scan the repository for the coverage XML file
        for file_path in Path(repo.path).rglob('*'):
            if 'coverage' in str(file_path) and file_path.suffix == '.xml':
                # Try to parse the coverage XML file
                try:
                    attributes = ElementTree.parse(file_path).getroot().attrib
                except ElementTree.ParseError as e:
                    raise ValueError(f'Error parsing coverage XML file: {file_path}') from e
                # Extract the coverage results
                del attributes['timestamp']
                del attributes['version']
                for key, value in attributes.items():
                    if key not in coverage_results:
                        coverage_results[key] = []
                    coverage_results[key].append(float(value))

        # Aggregate the coverage results
        for metric, scores in coverage_results.copy().items():
            if metric in COVERAGE_MEANS:
                coverage_results[metric] = sum(scores) / len(scores)
            elif metric in COVERAGE_SUMS:
                coverage_results[metric] = float(sum(scores))
        return coverage_results

    def check_results_file(self, tool: str, repo: GitRepo, skip_existing: bool) -> tuple:
        """Check if the results file already exists, and return the path if it does.

        :param str tool: The tool to check the results file for.
        :param GitRepo repo: The Git repository object.
        :param bool skip_existing: Whether to skip existing results.
        :return tuple: The results file, frequencies, and temporary file.
        """
        results_file, frequencies = None, None
        # Temporary output file. Deleted later.
        temp_file: str | Path = f'{tool}_{repo.name}.json'

        # File to save formatted results
        if self.output_dir:
            results_file = Path(self.output_dir, tool, temp_file)
            # Load the output from a file if possible
            if skip_existing and results_file and results_file.exists():
                logger.debug(f'Output file already exists: {results_file}')
                frequencies = json.loads(results_file.read_text(encoding='utf-8'))

        return results_file, frequencies, Path(repo.path, temp_file)

    def parse_results_file(self, tool: str, results_file: Path, temp_file: str | Path) -> dict:
        """Parse the results file from the specified tool.

        :param str tool: The tool to parse the results file for.
        :param Path results_file: The path to the results file.
        :param str | Path temp_file: The temporary file to parse.
        :return dict: The frequencies of the tool errors.
        """
        # Load the output from the temporary file
        with open(temp_file, 'r', encoding='utf-8') as f:
            output = json.load(f)

        # Count message frequencies
        frequencies = self.parse_json(tool, output)
        logger.debug(f'{tool}: {frequencies}')

        # Save results to a file if an output directory is provided
        if self.output_dir and results_file:
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(frequencies, f, indent=4)
            logger.info(f'{tool} output saved to {results_file}')

        # Delete the temporary file
        if Path(temp_file).is_file():
            os.remove(temp_file)
        return frequencies

    def parse_json(self, tool: str, output: dict) -> dict:
        """Parse the JSON output from the specified tool.

        :param str tool: The tool to parse the JSON output for.
        :param dict output: The JSON output from the tool.
        :return dict: The parsed results from the JSON output.
        """
        if tool == 'bandit':
            results = self.parse_json_bandit(output)
        elif tool == 'flake8':
            results = self.parse_json_flake8(output)
        elif tool == 'prospector':
            results = self.parse_prospector(output)
        elif tool == 'pylint':
            results = self.parse_pylint(output)
        elif tool == 'radon-cc':
            results = self.parse_radon_cyclomatic(output)
        elif tool == 'radon-hal':
            results = self.parse_radon_halstead(output)
        elif tool == 'radon-mi':
            results = self.parse_radon_maintainability(output)
        elif tool == 'radon-raw':
            results = self.parse_radon_raw(output)
        elif tool == 'ruff':
            results = self.parse_ruff(output)
        else:
            # logger.error(f'output: {json.dumps(output, indent=4)}')
            raise NotImplementedError(f'Parsing for {tool} not implemented')
        return self.format_results(tool, results)

    def parse_json_bandit(self, output_json: dict) -> dict:
        """Parse the output of bandit to get the frequencies of the message types.

        :param dict bandit_output: The output of running 'bandit'.
        :return dict: The frequencies of the message types.
        """
        # Overall statistics
        results = output_json['metrics']['_totals']
        # Frequencies of each message type
        for issue in output_json['results']:
            test_id = issue['test_id']
            results[test_id] = results.get(test_id, 0.0) + 1.0
        return results

    def parse_json_flake8(self, output_json: dict) -> dict:
        """Parse the output of flake8 to get the frequencies of the message types.

        :param dict output_json: The output of running 'flake8'.
        :return dict: The frequencies of the message types.
        """
        results: dict[str, float] = {}
        for errors in output_json.values():
            for error in errors:
                code = error.get('code', None)
                if code is not None:
                    results[code] = error.get(code, 0.0) + 1.0
        return results

    def parse_prospector(self, output_json: dict) -> dict:
        """Parse the output of prospector to get the frequencies of the message types.

        :param dict output_json: The output of running 'prospector'.
        :return dict: The frequencies of the message types.
        """
        results: dict[str, float] = {}
        for message in output_json['messages']:
            key = f"{message['source']}_{message['code']}"
            results[key] = results.get(key, 0.0) + 1.0
        return results

    def parse_pylint(self, output_json: dict) -> dict:
        """Parse the output of pylint to get the frequencies of the message types.

        :param dict output_json: The output of running 'pylint'.
        :return dict: The frequencies of the message types.
        """
        # Overall statistics
        results = output_json['statistics']['messageTypeCount']
        # Overall score
        results['score'] = output_json['statistics']['score']
        # Frequencies of each message type. UPDATE: removed as redundant
        # for message in output_json['messages']:
        #     message_id = message['messageId']
        #     results[message_id] = results.get(message_id, 0.0) + 1.0
        return results

    def parse_radon_cyclomatic(self, output_json: dict) -> dict:
        """Parse radon output to get cyclomatic complexity metrics.

        :param dict output_json: The output of running 'radon cc'.
        """
        # Extract the complexity scores from the output
        scores: dict[str, list[float]] = {}
        for parsed_file in output_json.values():
            for element in parsed_file:
                if 'complexity' in element:
                    if element['type'] not in scores:
                        scores[element['type']] = []
                    scores[element['type']].append(element['complexity'])
        # Calculate the min, max, mean, std, and var of the complexity scores
        results = {}
        for key, value in scores.items():
            results[f'min-cc_{key}'] = min(value)
            results[f'max-cc_{key}'] = max(value)
            results[f'mean-cc_{key}'] = sum(value) / len(value)
            results[f'std-cc_{key}'] = pd.Series(value).std()
            results[f'var-cc_{key}'] = pd.Series(value).var()
            results[f'sum-cc_{key}'] = sum(value)
        return results

    def parse_radon_halstead(self, output_json: dict) -> dict:
        """Parse radon output to get Halstead complexity metrics.

        :param dict output_json: The output of running 'radon hal'.
        :return dict: The sum and mean of each Halstead complexity metric.
        """
        totals = []
        for result in output_json.values():
            totals.append(result.get('total', {}))
        results = pd.DataFrame(totals)
        return {
            **{f'Mean Halstead {k.title()}': v for k, v in results.mean().to_dict().items()},
            **{f'Total Halstead {k.title()}': v for k, v in results.sum().to_dict().items()},
        }

    def parse_radon_maintainability(self, output_json: dict) -> dict:
        """Parse radon output to get maintainability index scores.

        :param dict output_json: The output of running 'radon mi'.
        :return dict: The number of files and mean maintainability index.
        """
        scores = [result['mi'] for result in output_json.values() if 'mi' in result]
        return {
            'Num. Files': len(output_json),
            'Mean Maintainability Index': sum(scores) / len(scores),
        }

    def parse_radon_raw(self, output_json: dict) -> dict:
        """Parse radon output to get raw analysis results.

        :param dict output_json: The output of running 'radon raw'.
        :return dict: The frequencies of the raw analysis results.
        """
        metric_names = [
            'LoC',  # number of lines of code
            'LLoC',  # number of logical lines of code
            'SLoC',  # number of source lines of code
            'Comments',
            'Multi-Line String Lines',  # Number of lines representing multi-line strings
            'Blank Lines',  # blank or whitespace-only lines
            'Single Line Comments',
        ]
        frequencies = {m: 0.0 for m in metric_names}
        for result in output_json.values():
            for metric, value in result.items():
                if metric in frequencies:
                    frequencies[metric] += float(value)
                else:
                    logger.debug(f'Ignoring unknown metric: {metric}')
        return frequencies

    def parse_ruff(self, output_json: dict) -> dict:
        """Parse ruff output to get the frequencies of the message types.

        :param dict output_json: The output of running 'ruff'.
        :return dict: The frequencies of the message types.
        """
        results: dict[str, int] = {}
        for error in output_json:
            error_code = error['code'] if error['code'] is not None else error['name']
            results[error_code] = error['count']
        return results

    def format_results(self, tool: str, results: dict) -> dict[str, float]:
        """Format the results of the analysis. Remove commas and convert to floats.

        :param dict results: The results of the analysis.
        :param str tool: The tool used to generate the results.
        :raises TypeError: If the results cannot be formatted.
        :return dict: The formatted results of the analysis.
        """
        try:  # Remove commas (for CSV compatibility) and convert to floats
            results = {k.replace(',', ''): float(v) for k, v in results.items()}
        except TypeError as e:
            logger.error(f'\n{json.dumps(results, indent=4)}\n')
            logger.error(f'Error parsing {tool} output: {e}')
            raise e
        return results

    def git_analysis(self, repo: GitRepo, verbose: bool = False) -> dict:
        """Analyze the Git repository.

        :param GitRepo repo: The Git repository object.
        :param bool verbose: Whether to include additional information, defaults to False.
        :return dict: The analysis of the Git repository.
        """
        logger.debug(f'Git analysis of {repo.path}...')
        analysis: dict[str, Any] = {
            # 'Repository': repo.name, 'path': repo.path,
            'LoC': repo.lines_of_code,
            'Num. Commits': repo.commit_count,
            'Num. Contributors': repo.num_contributors,
        }
        if verbose:
            analysis = {
                **analysis,
                'Branches': ', '.join(repo.get_branches()),
                'Contributors': ', '.join(repo.get_contributors()),
                'Latest Commit': repo.get_latest_commit(),
            }
        return analysis

    def parse_sonar_scanner_json(
        self, repo: GitRepo, output_dir: str | Path | None, skip_existing: bool = True
    ) -> dict:
        """Parse the JSON output from SonarScanner.

        :param GitRepo repo: The Git repository object.
        :param str | Path | None output_dir: Output directory, defaults to None.
        :param bool skip_existing: Whether to skip existing results, defaults to True.
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
        if skip_existing and parsed_measures_file.exists():
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
            if 'component' in line:
                measures = json.loads(line)['component']['measures']
                results = self.parse_sonar_scanner_line(measures, results)
            elif 'errors' in line:
                results['errors'] = len(json.loads(line)['errors'])
            else:
                raise ValueError(f'Could not parse line: {line}')

        # Save the parsed measures to a file
        parsed_measures_file.parent.mkdir(parents=True, exist_ok=True)
        with open(parsed_measures_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        return results

    def parse_sonar_scanner_line(
        self, measures: list[dict], results: dict[str, float | int]
    ) -> dict:
        """Parse a line of measures from SonarScanner output.

        :param list[dict] measures: The measures from SonarScanner output.
        :param dict[str, float | int] results: The results to update.
        :raises ValueError: If the measure cannot be parsed.
        :return dict: The updated results.
        """
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
            elif 'value' not in measure['metric'] and 'periods' in measure:
                results[measure['metric']] = float(len(measure['periods']))
            else:
                results[measure['metric']] = float(measure['value'])
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

        :param str | Path output_dir: Path to save the analysis
        :return pd.DataFrame df_results: The results as a pandas DataFrame.
        """
        if self.output_dir is None:
            raise ValueError('No output directory specified')

        # Save results to a CSV file
        results_file = Path(output_dir, 'results.csv')
        self.df_results = pd.DataFrame(self.results)
        self.df_results.to_csv(results_file, index=False)
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
        # if len(self.df_results) > 1:
        #     path = str(Path(self.output_dir, 'correlation_heatmap.csv'))
        #     Utils.save_heatmap(self.df_results, path, None, columns='all')

        # Summarize results and export to CSV, Markdown, and LaTeX
        self.summarize_tools(output_dir)
        return self.df_results

    def summarize_tools(self, output_dir: str | Path):
        """Summarize the tool metrics used in the analysis. Save as CSV, Markdown and LaTeX.

        :param str | Path output_dir: Path to save the analysis.
        """
        # Group keys by tool
        keys_by_tool: dict[str, list[str]] = {}
        for col in self.df_results.columns:
            if '__' in col:
                tool = col.split('__')[0]
                keys_by_tool[tool] = keys_by_tool.get(tool, []) + [col]

        # Bandit score = sum of all bandit issues
        metric_keys = [key for key in keys_by_tool['bandit'] if key.startswith('bandit__B')]
        self.df_results['bandit__Score'] = self.df_results[metric_keys].sum(axis=1)
        # Bandit score per line = Bandit score / LoC
        self.df_results['bandit__Score per Line'] = (
            self.df_results['bandit__Score'] / self.df_results['bandit__loc']
        )
        self.df_summary = self.df_results.drop(columns=metric_keys)

        # Summarize pylint metrics
        kept_keys = [
            f'pylint__{k}'
            for k in ['fatal', 'error', 'warning', 'refactor', 'convention', 'info', 'score']
        ]
        metric_keys = [k for k in keys_by_tool['pylint'] if k not in kept_keys]
        self.df_summary.drop(columns=metric_keys, inplace=True)

        # Rename/filter Radon metrics
        names = {
            f'radon-cc__mean-cc_{k}': f'radon-cc__Mean {k.capitalize()} CC'
            for k in ['class', 'function', 'method']
        }
        self.df_summary.rename(columns=names, errors='raise', inplace=True)
        self.df_summary.drop(columns=keys_by_tool['radon-cc'], inplace=True, errors='ignore')

        # Set index, sort columns, drop path column
        self.df_summary.set_index('name', drop=True, inplace=True)
        self.df_summary = self.df_summary[
            sorted([c for c in self.df_summary.columns if c != 'path'])
        ]

        # Save as CSV and Markdown
        filename = Path(output_dir, 'summary.csv')
        self.df_summary.to_csv(filename, index=True, float_format='%.2f')
        logger.info(f'Saved summary to {filename}')
        self.df_summary.round(2).to_markdown(filename.with_suffix('.md'))

        # Format and group columns
        def split_key(column):
            parts = column.replace('_', ' ').split('  ')
            metric = parts[-1].title() if parts[-1] == parts[-1].lower() else parts[-1]
            metric = metric.replace('Loc', 'LoC').replace('Nosec', 'Nosec Comments')
            return parts[0].title(), metric

        self.df_summary.columns = pd.MultiIndex.from_tuples(
            [split_key(c) for c in sorted(self.df_summary.columns.tolist())],
            names=['Tool', 'Metric'],
        )
        self.df_summary.round(2).style.to_latex(filename.with_suffix('.tex'))  # Save as LaTeX
