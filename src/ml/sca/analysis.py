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
from ml.sca.repo import GitRepo
from ml.sca.reporting import Reporting

IGNORED_COLS: dict[str, list[str]] = {
    'coverage': [],
    'prospector': ['dodgy Score'],  # Missing most scores
    'sonar': [
        'branch_coverage',  # almost identical to coverage's Branch Rate
        'conditions_to_cover',  # almost identical to coverage's Branches Valid
        'lines_to_cover',  # almost identical to coverage's Lines Valid
        'line_coverage',  # almost identical to SonarQube's coverage
        'alert_status',
        'last_commit_date',  # irrelevant
        'ncloc_language_distribution',  # Only Python examined here
        'quality_gate_details',  # Specific to SonarQube
        'quality_profiles',  # Specific to SonarQube
    ],
}


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
        if self.is_git_repo(self.input_dir):
            # Check if the directory is a Git repository
            self.results = [self.analyze_repo(self.input_dir)]
        else:
            # Otherwise, assume the directory contains Git repositories
            repos = [d for d in self.input_dir.iterdir() if self.is_git_repo(d)]
            if len(repos) == 0:
                raise ValueError(f'No Git repositories found in directory: {self.input_dir}')
            self.results = [self.analyze_repo(repo) for repo in repos]

        # Save the results to a file if an output directory is provided
        if self.output_dir:
            self.df_results = pd.DataFrame(self.results)
            # self.df_results = pd.read_csv(Path(self.output_dir, 'SUMMARY', 'csv', 'results.csv'))
            Reporting.save_results(self.df_results, self.output_dir, IGNORED_COLS)
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
        if target_dir is None:
            raise ValueError('No target directory specified for analysis')

        logger.info(f'Analyzing {target_dir}...')
        repo = GitRepo.from_package_name(str(Path(target_dir).name), target_path=target_dir)

        # Run Coverage, git and Sonar analysis on the repository
        sonar_results = self._parse_sonar_scanner_json(repo, self.output_dir, skip_existing_sonar)
        results = {
            'name': repo.library.git_name,
            'path': repo.path,
            **{f'coverage__{k}': v for k, v in self._read_coverage_xml(repo).items()},
            # **{f'git__{k}': v for k, v in self.git_analysis(repo, verbose=False).items()},
            **{f'sonar__{k}': v for k, v in sonar_results.items()},
        }

        # Run other (CLI) tools on the repository
        for tool, command in self.build_commands(repo.library.git_name).items():
            outputs = self._run_cli_command(tool, command, repo, skip_existing_sca)
            results.update({f'{tool}__{k}': v for k, v in outputs.items()})
        return results

    def build_commands(self, repo_name: str | Path) -> dict:
        """Build the CLI commands for the static code analysis tools.

        :param str | Path repo_name: The name of the repository.
        :return dict: The CLI commands for the static code analysis tools.
        """
        filename = f'_{repo_name}.json'
        # fmt: off
        commands = {
            'bandit': [
                'bandit', '-r', '--exclude', 'tests,unittests,test,docs',
                '--format', 'json', '-o', f'bandit{filename}', '.'
            ],
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

    def _run_cli_command(
        self,
        tool: str,
        command: list,
        repo: GitRepo,
        skip_existing: bool,
        verbose: bool = True,
    ) -> dict:
        """Run a CLI command on the specified directory.

        :param str tool: The tool to run the CLI command for.
        :param list command: The CLI command to run.
        :param GitRepo repo: The Git repository object.
        :param bool skip_existing: Whether to skip existing results.
        :param bool verbose: Whether to include additional information, defaults to False.
        :raises ValueError: If the tool is not supported.
        :return dict: The frequencies of the tool errors.
        """
        # Check if the results file already exists
        result_path, frequencies, json_file = self._check_results_file(tool, repo, skip_existing)

        # Return the frequencies if they already exist
        if frequencies and skip_existing:
            return frequencies
        logger.debug(f'Running {tool}...')

        # Avoid library specific configuration error
        if repo.name == 'auto_sklearn':
            command[-1] += '/auto-sklearn'

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
        frequencies = self._parse_results_file(tool, result_path, json_file)
        return frequencies

    def _read_coverage_xml(self, repo: GitRepo):
        """Read the coverage XML file from the repository if present.

        :param GitRepo repo: The Git repository object.
        :param str pattern: The pattern to search for, defaults to 'coverage.xml'.
        """
        logger.info(f'Reading coverage XML file for {repo.name}...')
        coverage_results: dict[str, float] = {}
        # Scan the repository for the coverage XML file
        for file_path in Path(repo.path).rglob('*'):
            if 'coverage' in str(file_path) and file_path.suffix == '.xml':
                # Try to parse the coverage XML file
                try:
                    attributes = ElementTree.parse(file_path).getroot().attrib  # type: ignore
                except ElementTree.ParseError as e:
                    raise ValueError(f'Error parsing coverage XML file: {file_path}') from e
                # Extract the coverage results
                del attributes['timestamp']
                del attributes['version']
                for key, value in attributes.items():
                    if key in coverage_results:
                        raise KeyError(f'Key "{key}" already exists in coverage results')
                    coverage_results[key] = float(value)
        return coverage_results

    def _check_results_file(self, tool: str, repo: GitRepo, skip_existing: bool) -> tuple:
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
                # logger.debug(f'Output file already exists: {results_file}')
                frequencies = json.loads(results_file.read_text(encoding='utf-8'))

        return results_file, frequencies, Path(repo.path, temp_file)

    def _parse_results_file(self, tool: str, results_file: Path, temp_file: str | Path) -> dict:
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
        frequencies = self._parse_json(tool, output)
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

    def _parse_json(self, tool: str, output: dict) -> dict:
        """Parse the JSON output from the specified tool.

        :param str tool: The tool to parse the JSON output for.
        :param dict output: The JSON output from the tool.
        :return dict: The parsed results from the JSON output.
        """
        if tool == 'bandit':
            results = self._parse_json_bandit(output)
        elif tool == 'flake8':
            results = self._parse_json_flake8(output)
        elif tool == 'prospector':
            results = self._parse_prospector(output)
        elif tool == 'pylint':
            results = self._parse_pylint(output)
        elif tool == 'radon-cc':
            results = self._parse_radon_cyclomatic(output)
        elif tool == 'radon-hal':
            results = self._parse_radon_halstead(output)
        elif tool == 'radon-mi':
            results = self._parse_radon_maintainability(output)
        elif tool == 'radon-raw':
            results = self._parse_radon_raw(output)
        elif tool == 'ruff':
            results = self._parse_ruff(output)
        else:
            # logger.error(f'output: {json.dumps(output, indent=4)}')
            raise NotImplementedError(f'Parsing for {tool} not implemented')
        return self.format_results(tool, results)

    def _parse_json_bandit(self, output_json: dict) -> dict:
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

    def _parse_json_flake8(self, output_json: dict) -> dict:
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

    def _parse_prospector(self, output_json: dict) -> dict:
        """Parse the output of prospector to get the frequencies of the message types.

        :param dict output_json: The output of running 'prospector'.
        :return dict: The frequencies of the message types.
        """
        results: dict[str, float] = {}
        results['Prospector Num. Issues'] = output_json['summary']['message_count']
        # Count errors by tool and message type
        for message in output_json['messages']:
            # Count errors by tool
            tool = f'{message["source"].title()} Num. Issues'
            results[tool] = results.get(tool, 0.0) + 1.0
            # Count the frequencies of each message type
            tool_error = f"{message['source']}_{message['code']}"
            results[tool_error] = results.get(tool_error, 0.0) + 1.0
        return results

    def _parse_pylint(self, output_json: dict) -> dict:
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

    def _parse_radon_cyclomatic(self, output_json: dict) -> dict:
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
        # Calculate the std and sum of complexity scores
        results = {}
        for key, value in scores.items():
            results[f'mean-cc_{key}'] = sum(value) / len(value)
            results[f'total-cc_{key}'] = sum(value)
        return results

    def _parse_radon_halstead(self, output_json: dict) -> dict:
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

    def _parse_radon_maintainability(self, output_json: dict) -> dict:
        """Parse radon output to get maintainability index scores.

        :param dict output_json: The output of running 'radon mi'.
        :return dict: The number of files and mean maintainability index.
        """
        scores = [result['mi'] for result in output_json.values() if 'mi' in result]
        return {
            'Num. Files': len(output_json),
            'Mean Maintainability Index': sum(scores) / len(scores),
        }

    def _parse_radon_raw(self, output_json: dict) -> dict:
        """Parse radon output to get raw analysis results.

        :param dict output_json: The output of running 'radon raw'.
        :return dict: The frequencies of the raw analysis results.
        """
        metric_names = {
            'loc': 'LoC',  # number of lines of code
            'lloc': 'LLoC',  # number of logical lines of code
            'sloc': 'SLoC',  # number of source lines of code
            'comments': 'Comments',
            'multi': 'Multi-Line String Lines',  # Number of lines representing multi-line strings
            'blank': 'Blank Lines',  # blank or whitespace-only lines
            'single_comments': 'Single Line Comments',
        }
        frequencies = {m: 0.0 for m in metric_names.values()}
        for result in output_json.values():
            for metric, value in result.items():
                if metric == 'error':
                    continue
                frequencies[metric_names[metric]] += float(value)
        return frequencies

    def _parse_ruff(self, output_json: dict) -> dict:
        """Parse ruff output to get the frequencies of the message types.

        :param dict output_json: The output of running 'ruff'.
        :return dict: The frequencies of the message types.
        """
        results: dict[str, int] = {}
        for error in output_json:
            error_code = error['code'] if error['code'] is not None else error['name']
            results[error_code] = error['count']
        results['ruff Num. Issues'] = sum(results.values())
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

    def _git_analysis(self, repo: GitRepo, verbose: bool = False) -> dict:
        """Analyze the Git repository.

        :param GitRepo repo: The Git repository object.
        :param bool verbose: Whether to include additional information, defaults to False.
        :return dict: The analysis of the Git repository.
        """
        logger.debug(f'Git analysis of {repo.path}...')
        analysis: dict[str, Any] = {'LoC': repo.lines_of_code, 'Num. Commits': repo.commit_count}
        if verbose:
            analysis = {
                **analysis,
                'Branches': ', '.join(repo.get_branches()),
                'Contributors': ', '.join(repo._get_contributors()),
                'Latest Commit': repo._get_latest_commit(),
            }
        return analysis

    def _parse_sonar_scanner_json(
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

        sonar_dir = Path(output_dir, 'sonar', Path(repo.path).name)
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
                results = self._parse_sonar_scanner_line(measures, results)
            elif 'errors' in line:
                results['errors'] = len(json.loads(line)['errors'])
            else:
                raise ValueError(f'Could not parse line: {line}')

        # Save the parsed measures to a file
        parsed_measures_file.parent.mkdir(parents=True, exist_ok=True)
        with open(parsed_measures_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)

        return results

    def _parse_sonar_scanner_line(
        self, measures: list[dict], results: dict[str, float | int]
    ) -> dict:
        """Parse a line of measures from SonarScanner output.

        :param list[dict] measures: The measures from SonarScanner output.
        :param dict[str, float | int] results: The results to update.
        :raises ValueError: If the measure cannot be parsed.
        :return dict: The updated results.
        """
        # Filter out ignored metrics
        measures = [m for m in measures if m['metric'] not in IGNORED_COLS['sonar']]

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
