"""Call the main function of the module from the command line."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource

from ml import Library
from ml.logs import Logs


class LogLevelEnum(str, Enum):
    """Log level options."""

    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'


# Default configuration settings
SETTINGS: dict = {
    'cpu_only': {
        'default': True,
        'description': 'Use CPU only, even if GPU is available.',
    },
    'dataset': {
        'default': None,
        'description': 'Named dataset or path to a file/directory.',
    },
    'data_dir': {
        'default': 'data',
        'description': 'Path to the data directory to store datasets.',
    },
    'libraries': {
        'default': ['none'],
        'description': 'List of libraries to use',
        'options': ['all', 'none'] + [lib.value for lib in Library],
    },
    'log_file': {
        'default': None,
        'description': 'Log file path. If None, logs will not be saved to a file.',
    },
    'log_level': {
        'default': LogLevelEnum.DEBUG,
        'description': 'Logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.',
    },
    'n_jobs': {
        'default': 1,
        'description': 'Number of parallel jobs',
    },
    'output_dir': {
        'default': 'results',
        'description': 'Output directory.',
    },
    'preprocessed_subdir': {
        'default': 'preprocessed',
        'description': 'Subdirectory within data_dir to store preprocessed datasets.',
    },
    'random_state': {
        'default': 1,
        'description': 'Random state for reproducibility.',
    },
    'task': {
        'default': None,
        'description': 'Task type. If None, program will end.',
    },
    'verbosity': {
        'default': 1,
        'description': 'Verbosity level.',
    },
}


def cli_field(*args, **kwargs):
    """Create a field for the CLI with default settings."""
    return Field(*args, repr=True, validate_default=True, **kwargs)


class SettingsSource(CliSettingsSource):
    def _parse_args(self, parser, args):
        """Only consume known args; ignore the rest (e.g. pytest flags)."""
        known, _ = parser.parse_known_args(args)
        return vars(known)


class Configuration(BaseSettings):
    """Application configuration.

    :param bool cpu_only: Use CPU only, even if GPU is available.
    :param str dataset: Name of dataset
    :param str | Path data_dir: Path to the data directory
    :param list[Library | str] libraries: AutoML libraries to run.
    :param str | Path log_file: Log file path. If None, logs will not be saved to a file.
    :param LogLevelEnum log_level: Logging level. Options: DEBUG, INFO, WARNING, ERROR, CRITICAL.
    :param int n_jobs: Number of processes to run.
    :param str | Path output_dir: Output directory for results and logs.
    :param str | Path preprocessed_subdir: Subdirectory in data_dir for preprocessed datasets.
    :param int random_state: Random seed for reproducibility.
    :param str | None task: Task type. If None, program will end.
    :param int verbosity: Verbosity level.
    """

    cpu_only: bool = cli_field(**SETTINGS['cpu_only'])
    dataset: str | None = cli_field(**SETTINGS['dataset'])
    data_dir: str | Path = cli_field(**SETTINGS['data_dir'])
    libraries: list[Library | str] = cli_field(**SETTINGS['libraries'])
    log_file: str | Path | None = cli_field(**SETTINGS['log_file'])
    log_level: LogLevelEnum = cli_field(**SETTINGS['log_level'])
    n_jobs: int = cli_field(**SETTINGS['n_jobs'])
    output_dir: str | Path = cli_field(**SETTINGS['output_dir'])
    preprocessed_subdir: str | Path = cli_field(**SETTINGS['preprocessed_subdir'])
    random_state: int = cli_field(**SETTINGS['random_state'])
    task: str | None = cli_field(**SETTINGS['task'])
    verbosity: int = cli_field(**SETTINGS['verbosity'])

    # Pydantic parameters
    model_config = {
        'cli_ignore_unknown_args': True,
        'cli_parse_args': True,
        'cli_settings_source': SettingsSource,  # Used to ignore unknown CLI arguments
        'env_prefix': '',  # Prefix for environment variables
        'extra': 'allow',  # Extra CLI arguments
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Set up logging
        Logs.log_to_stderr()
        if self.log_file:
            Logs.log_to_file(sink=str(self.log_file))
        # Validate libraries
        self._validate_libraries()

    def _validate_libraries(self):
        """Validate the libraries list."""
        all_libs = [lib.value for lib in Library]
        if 'all' in self.libraries:
            self.libraries = all_libs
        elif 'none' in self.libraries or 'prepare_data' in self.libraries:
            self.libraries = []
        elif 'installed' in self.libraries:
            self.libraries = []
            for lib in Library:
                try:
                    __import__(lib.value)
                    self.libraries.append(lib.value)
                except ImportError:
                    logger.warning(f'Library "{lib.value}" is not installed.')
        else:
            for lib in self.libraries:
                if lib not in all_libs:
                    raise ValueError(f'Invalid library: "{lib}". Valid options are: {all_libs}')

    def log_configuration(self):
        """Log configuration."""
        logger.info(f'Configuration:\n{json.dumps(self.model_dump(), indent=2)}')
