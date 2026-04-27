"""Anomaly Detection module."""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger
import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, CliSettingsSource

from ml.ad.analysis import Analysis
from ml.ad.training import SKABTrainer


def cli_field(*args, **kwargs):
    """Create a field for the CLI with default settings."""
    return Field(*args, repr=True, validate_default=True, **kwargs)


class SettingsSource(CliSettingsSource):
    """Configure pydantic CLI to consume known args and ignore the rest."""

    def _parse_args(self, parser, args):  # pylint: disable=method-hidden
        """Only consume known args; ignore the rest (e.g. pytest flags)."""
        known, _ = parser.parse_known_args(args)
        return vars(known)


class CLIConfiguration(BaseSettings):
    """Application CLI configuration.

    :param float | None contamination: Contamination level for anomaly detection models.
    :param Path | str data_dir: Path to the data directory
    :param str plots: The plotting behaviour: "all", "none", "skip_existing".
    :param Path | str results_dir: Base directory to save results.
    :param bool skip_done_experiments: Whether to skip completed experiments.
    :param str tool: The anomaly detection tool to use (e.g. PyCaretADModel, TimeSeriesOD).
    :param int verbosity: Verbosity level for logging and library logging (0=ERROR, 1=INFO, 2=DEBUG, 3=TRACE).
    :param int | None window_size: Window size for feature engineering (if applicable).
    """

    contamination: float | str | None = cli_field(
        default=None, description='Contamination level for anomaly detection models.'
    )
    data_dir: Path | str = cli_field(
        default=Path('data/SKAB'), description='Path to the data directory.'
    )
    plots: str = cli_field(
        default='all', description='Plotting behaviour: "all", "none", "skip_existing".'
    )
    results_dir: Path | str = cli_field(
        default=Path('results/ad'), description='Base directory to save results.'
    )
    skip_done_experiments: bool = cli_field(
        default=True, description='Whether to skip completed experiments.'
    )
    test_set_size: float = cli_field(
        default=0.25, description='Proportion of data to use as test set (if applicable).'
    )
    tool: str = cli_field(
        default='PyCaretADModel',
        description='The anomaly detection tool to use (e.g. PyCaretADModel, TimeSeriesOD).',
    )
    verbosity: int = cli_field(
        default=2,
        description='Verbosity level for logging and library logging.',
    )
    window_size: int | str | None = cli_field(
        default='auto', description='Window size for feature engineering (if applicable).'
    )

    # Pydantic parameters
    model_config = {  # type: ignore
        'cli_ignore_unknown_args': True,
        'cli_parse_args': True,
        'cli_settings_source': SettingsSource,  # Used to ignore unknown CLI arguments
        'env_prefix': '',  # Prefix for environment variables
        'extra': 'allow',  # Extra CLI arguments
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._validate_contamination()
        self._validate_plots()
        self._validate_tool()
        self._validate_test_set_size()
        self._validate_verbosity()
        self._validate_window_size()

        # Validate data_dir
        if not isinstance(self.data_dir, (str, Path)):
            raise ValueError('data_dir must be a string or Path object.')
        if not Path(self.data_dir).exists():
            raise ValueError(f'data_dir "{self.data_dir}" does not exist.')

        # Validate results_dir
        if not isinstance(self.results_dir, (str, Path)):
            raise ValueError('results_dir must be a string or Path object.')

    def _validate_contamination(self):
        """Validate the contamination parameter."""
        if self.contamination == 'None':
            self.contamination = None
        elif self.contamination is not None:
            try:
                self.contamination = float(self.contamination)
            except ValueError as e:
                raise ValueError('Contamination must be a float, "None", or None.') from e
        if self.contamination is not None and not 0 < self.contamination < 1:
            raise ValueError('Contamination must be None or a float between 0 and 1.')

    def _validate_plots(self):
        """Validate the plots parameter."""
        valid_options = ['all', 'none', 'skip_existing']
        if isinstance(self.plots, str):
            self.plots = self.plots.lower().strip()
            if self.plots not in valid_options:
                raise ValueError(f'plots must be one of {valid_options}.')

    def _validate_test_set_size(self):
        """Validate the test_set_size parameter."""
        try:
            self.test_set_size = float(self.test_set_size)
        except ValueError as e:
            raise ValueError('test_set_size must be a float between 0 and 1.') from e
        if not 0 < self.test_set_size < 1:
            raise ValueError('test_set_size must be a float between 0 and 1.')

    def _validate_tool(self):
        """Validate the tool parameter and standardize its value."""
        if self.tool.lower().strip() == 'pycaret':
            self.tool = 'PyCaretADModel'
        elif self.tool.lower().strip() == 'ts_od':
            self.tool = 'TimeSeriesODModel'
        elif self.tool.lower().strip() == 'lunar':
            self.tool = 'LunarADModel'

        valid_tools = ['PyCaretADModel', 'TimeSeriesODModel', 'LunarADModel']
        if self.tool not in valid_tools:
            raise ValueError(f'tool must be one of {valid_tools}.')

    def _validate_verbosity(self):
        """Validate the verbosity parameter."""
        if not isinstance(self.verbosity, int) or self.verbosity not in range(0, 4):
            raise ValueError('Verbosity must be an integer between 0 and 3.')

    def _validate_window_size(self):
        """Validate the window_size parameter."""
        if self.window_size == 'None':
            self.window_size = None
        elif self.window_size == 'auto':
            if self.tool in ['TimeSeriesODModel', 'LunarADModel']:
                self.window_size = 50  # Default for time series models
            else:
                self.window_size = None  # AutoML should handle this
        else:
            try:
                self.window_size = int(self.window_size)  # type: ignore
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f'window_size must be a positive integer or None. Got: {self.window_size}'
                ) from e

        if self.window_size is not None and (
            not isinstance(self.window_size, int) or self.window_size <= 0
        ):
            raise ValueError(
                f'window_size must be a positive integer or None. Got: {self.window_size}'
            )


def load_skab(root_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the SKAB dataset.

    :param str | Path root_dir: Path to the SKAB 'data' directory.
    :return: A dictionary mapping 'subfolder/filename.csv' -> DataFrame.
    """
    dataframes = {}
    # Walk through all directories and files under the root directory
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.csv'):
                file_path = os.path.join(subdir, file)
                # Load CSV
                df = pd.read_csv(file_path, delimiter=';')
                # Normalise column names
                df.columns = [c.strip().lower() for c in df.columns]
                # Parse timestamp column if present
                for col in df.columns:
                    if 'datetime' in col or 'time' in col or 'timestamp' in col:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                # Dictionary key: relative path
                dataframes[os.path.relpath(file_path, root_dir)] = df
    return dataframes


def main():
    """Main function to run anomaly detection experiments."""
    configuration = CLIConfiguration()

    # Configure logging
    level_mapping = {0: 'ERROR', 1: 'INFO', 2: 'DEBUG', 3: 'TRACE'}
    logger.remove()  # Remove default logger
    logger.add(
        lambda msg: print(msg, end=''),  # Print to stdout without extra newline
        level=level_mapping.get(configuration.verbosity, 'INFO'),
        colorize=True,
        format=(
            '|<green>{time:YYYY-MM-DD HH:mm:ss}</green>|<level>{level: <8}</level>'
            '|<cyan>{file.path}</cyan>:<cyan>{line: <3}</cyan>| <level>{message}</level>'
        ),
    )
    logger.info(f'Configuration:\n{json.dumps(configuration.model_dump(mode="json"), indent=2)}')

    # Load SKAB dataset
    skab_data = load_skab(configuration.data_dir)
    logger.debug('Loaded SKAB datasets.')

    results_dir = Path(
        configuration.results_dir,
        configuration.tool,
        f'contamination_{configuration.contamination}',
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    trainer = SKABTrainer(
        tool=configuration.tool,
        contamination=configuration.contamination,
        results_dir=results_dir,
        skip_done_experiments=configuration.skip_done_experiments,
        test_set_size=configuration.test_set_size,
        verbosity=configuration.verbosity,
        window_size=configuration.window_size,
    )

    all_metadata = {}
    all_results = []
    # Iterate through datasets and run anomaly detection
    for name, df in skab_data.items():
        if not any(k in name for k in ['valve1', 'valve2']):
            continue
        logger.debug(f'Using dataset: "{name}", shape: {df.shape}')
        trainer.train_models(
            name=name, dataframes=skab_data, all_metadata=all_metadata, all_results=all_results
        )

    logger.info('Analyzing results...')
    analysis = Analysis(
        results_dir, plots=configuration.plots, test_set_size=configuration.test_set_size
    )
    analysis.analyse_results(all_results, all_metadata)
    logger.success(
        f'Completed experiments for contamination={configuration.contamination}. '
        f'Results saved to: {results_dir}'
    )


if __name__ == "__main__":
    main()
