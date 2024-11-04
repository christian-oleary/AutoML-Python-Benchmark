"""
Data formatting functions
"""

from abc import ABCMeta, abstractmethod
import argparse
from datetime import datetime
import os
from pathlib import Path
import math
from urllib.parse import urlparse
import zipfile

import pandas as pd
from pandas import DataFrame
from typing_extensions import Self

from src.automl.frequencies import frequencies
from src.automl.logs import logger
from src.automl.TSForecasting.data_loader import convert_tsf_to_dataframe, FREQUENCY_MAP
from src.automl.validation import Task


class Dataset(metaclass=ABCMeta):
    """Base class for datasets"""

    # Dataset aliases
    aliases: list[str] = ['_base_dataset']

    # Default data directory
    path: str | Path | None = None
    url: str | Path | None = None

    # Dataframes
    df: pd.DataFrame
    train_df: pd.DataFrame
    test_df: pd.DataFrame

    # Frequency and horizon
    frequency: str
    horizon: int

    # Start and end times for training and test sets
    train_set_start_time: str | None = None
    train_set_end_time: str | None = None
    test_set_start_time: str | None = None
    test_set_end_time: str | None = None

    # Task Type
    task: Task = Task.NONE

    def __init__(
        self,
        path: str | Path | None = None,
        url: str | None = None,
        init_dataset: bool = True,
        **kwargs,
    ):
        # Validate inputs
        if not path and not url:
            raise ValueError('No path or URL provided for dataset')

        if isinstance(path, (str, Path)):
            logger.info(f'Reading dataset from path: {path}')
            self.path = Path(path)
            if not self.path.exists():
                raise FileNotFoundError(f'Path not found: {self.path}')

        elif isinstance(url, str):
            self.url = url
            logger.info(f'Fetching dataset from URL: {url}')
            urlparse(self.url)  # Validate URL

        # Fetch data
        if init_dataset:
            self._init_dataset(**kwargs)
            self.split_data()

    @abstractmethod
    def _init_dataset(self, **kwargs) -> Self:
        """Fetch data relating to dataset."""
        self.frequency = kwargs.get('frequency', self.frequency)
        self.horizon = int(kwargs.get('horizon', self.horizon))

        self.train_set_start_time = kwargs.get('train_set_start_time', self.train_set_start_time)
        self.train_set_end_time = kwargs.get('train_set_end_time', self.train_set_end_time)
        self.test_set_start_time = kwargs.get('test_set_start_time', self.test_set_start_time)
        self.test_set_end_time = kwargs.get('test_set_end_time', self.test_set_end_time)

        self.task = kwargs.get('task', self.task)
        return self

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test sets."""
        self.ensure_data()
        # Holdout for model testing (80% training, 20% testing).
        # This seems to be used by Godahewa et al. for global forecasting:
        # https://github.com/rakshitha123/TSForecasting/blob/master/experiments/rolling_origin.R#L10
        # Also used by Bauer 2021 for univariate forecasting
        self.train_df = self.df.head(int(len(self.df) * 0.8))
        self.test_df = self.df.tail(int(len(self.df) * 0.2))
        return self.train_df, self.test_df

    def ensure_data(self):
        """Ensure dataset is not empty.

        :raises ValueError: If dataset is empty
        """
        if self.df is None or len(self.df) == 0:
            raise ValueError('Empty dataset!')

    @classmethod
    def load_dataset(cls, name: str, path_or_url: str | Path | None = None, **kwargs) -> Self:
        """Load a dataset by name or file path

        :param str name: Dataset name
        :param str | Path | None path_or_url: Dataset path, defaults to None
        :param kwargs: Additional arguments for dataset loading
        :return Dataset: Dataset object
        """
        logger.debug(f'Loading dataset "{name}" from {path_or_url}')
        if not name:
            raise ValueError(f'No dataset name provided: {name}')

        # Attempt to fetch dataset by name
        if path_or_url is None:
            name = str(path_or_url).lower().strip()
            dataset = None
            for ds in cls.__subclasses__():
                if name in [ds.__name__.lower(), *ds.aliases]:
                    dataset = ds(**kwargs)
                    break

        # Attempt to fetch dataset from URL
        elif str(path_or_url).lower().startswith('http'):
            raise NotImplementedError('URL loading not implemented')

        # Attempt to read dataset from file path
        else:
            path = Path(path_or_url)
            if not path.exists():
                raise FileNotFoundError(f'Path not found: {path}')

        if dataset is None:
            raise ValueError(f'Unknown dataset: {name}')
        return dataset


class ISEMDataset(Dataset):
    """I-SEM dataset"""

    aliases: list[str] = ['isem']
    frequency: str = '24H'
    horizon: int = 24
    has_nans: bool = False

    def _init_dataset(self, **kwargs) -> Self:
        """Fetch I-SEM data.

        :raises ValueError: If no path provided
        """
        super()._init_dataset(**kwargs)
        if not self.path:
            raise ValueError('No path provided for I-SEM dataset')

        # Read I-SEM data. Expecting 'applicable_date' column to use as index
        self.df = pd.read_csv(self.path)
        if 'applicable_date' not in self.df.columns:
            raise ValueError('Missing applicable_date column')
        self.df = self.df.set_index('applicable_date')
        return self


class ISEM2020Dataset(ISEMDataset):
    """I-SEM 2020 dataset"""

    aliases: list[str] = ['isem2020']
    train_set_start_time: str = '2019/12/31 23:00:00'
    train_set_end_time: str = '2020/10/19 23:00:00'
    test_set_start_time: str = '2020/10/20 00:00:00'
    test_set_end_time: str = '2020/12/31 22:00:00'

    def split_data(self) -> tuple[DataFrame, DataFrame]:
        """Split data into training and test sets."""
        self.ensure_data()
        logger.debug('Loading I-SEM 2020 dataset')
        # Ensure index is datetime
        self.df.index = pd.to_datetime(self.df.index)
        self.train_df = self.df.loc[self.train_set_start_time : self.train_set_end_time, :]  # type: ignore
        self.test_df = self.df.loc[self.test_set_start_time : self.test_set_end_time, :]  # type: ignore
        return self.train_df, self.test_df


# class LibraDataset(Dataset):
#     self.data = metadata[metadata['file'] == csv_file]
#     self.frequency = int(self.data['frequency'].iloc[0])
#     self.horizon = int(self.data['horizon'].iloc[0])
#     self.actual = self.test_df.copy().values.flatten()
#     self.y_train = self.train_df.copy().values.flatten()  # Required for MASE
#     # Libra's custom rolling origin forecast:
#     # kwargs = {
#     #     'origin_index': int(self.data['origin_index'].iloc[0]),
#     #     'step_size': int(self.data['step_size'].iloc[0])
#     #     }
# self.df = pd.read_csv(self.dataset_path, header=None)
# self.df.columns = ['target']


# class MonashDataset(Dataset):
# # Filter datasets based on "Monash Time Series Forecasting Archive" by Godahewa et al. (2021)
# # we do not consider the London smart meters, wind farms, solar power, and wind power datasets
# # for both univariate and global model evaluations, the Kaggle web traffic daily dataset for
# # the global model evaluations and the solar 10 minutely dataset for the WaveNet evaluation
# filter_forecast_datasets = True  # TODO: make an env variable
# if filter_forecast_datasets and self.dataset_name in self.omitted_datasets:
#     logger.debug(f'Skipping dataset {self.dataset_name}')
#     continue

#     # self.data = metadata[metadata['file'] == csv_file.replace('csv', 'tsf')]
#     # self.frequency = self.data['frequency'].iloc[0]
#     # self.horizon = self.data['horizon'].iloc[0]
#     # if pd.isna(self.horizon):
#     #     raise ValueError(f'Missing horizon in 0_metadata.csv for {csv_file}')
#     # self.horizon = int(self.horizon)
#     # # TODO: revise frequencies, determine and data formatting stage
#     # if pd.isna(self.frequency) and 'm3_other_dataset.csv' in csv_file:
#     #     self.frequency = 'yearly'
#     # self.actual = self.test_df.values

#     self.df = pd.read_csv(self.dataset_path, index_col=0)


class DatasetFormatter:
    """Methods for formatting raw datasets in preparation for modelling."""

    default_start_timestamp = datetime.strptime('1970-01-01 00-00-00', '%Y-%m-%d %H-%M-%S')

    def format_data(self, config: argparse.Namespace):
        """Format data in a given config.data_dir in preparation for modelling

        :param argparse.Namespace config: arguments from command line
        """
        if config.task == Task.UNIVARIATE_FORECASTING.value:
            self.format_univariate_forecasting_data(config.data_dir)
        elif config.task == Task.GLOBAL_FORECASTING.value:
            self.format_global_forecasting_data(config.data_dir)
        elif config.task == Task.NONE:
            pass
        else:
            raise NotImplementedError()

    def format_univariate_forecasting_data(self, data_dir: str | Path) -> None:
        """Format data for univariate forecasting

        :param str data_dir: Data directory
        """
        logger.info('Reading univariate forecasting data...')
        meta_data: dict = {
            'file': [],
            'horizon': [],
            'frequency': [],
            'nan_count': [],
            'num_rows': [],
            'num_cols': [],
        }

        headers_and_timestamps = 'libra' not in str(data_dir)
        if not headers_and_timestamps:  # I-SEM data
            meta_data['origin_index'] = []
            meta_data['step_size'] = []

        csv_files = [
            f for f in os.listdir(data_dir) if '0_metadata.csv' not in f and f.endswith('csv')
        ]

        for csv_file in csv_files:

            # Read data into a DataFrame
            csv_path = os.path.join(data_dir, csv_file)
            if headers_and_timestamps:  # I-SEM data
                df = pd.read_csv(csv_path)
                df = df.set_index('applicable_date')
                if df.shape[1] != 1:
                    raise AssertionError(f'Expected one column. Shape: {df.shape}')

                meta_data['file'].append(csv_file)
                meta_data['horizon'].append(24)  # hourly data
                meta_data['frequency'].append(24)  # hourly data
                meta_data['nan_count'].append(int(df.isna().sum().values.sum()))
                meta_data['num_rows'].append(df.shape[0])
                meta_data['num_cols'].append(df.shape[1])

            else:  # Libra dataset
                df = pd.read_csv(os.path.join(data_dir, csv_file), header=None)
                if df.shape[1] != 1:
                    raise AssertionError(f'Expected one column. Shape: {df.shape}')

                # The horizon/frequency are based on the paper:
                # "Libra: A Benchmark for Time Series Forecasting Methods" Bauer 2021
                # - "the horizon is 20% of the time series length"
                # - "the [rolling origin] starting point is set either to [a maximum of] 40% of
                #    the time series or at two times the frequency of the time series plus 1"
                # - "the range between the starting point and endpoint is divided into 100 [equal
                #    (rounded up)] parts"
                frequency = frequencies[csv_file]
                meta_data['file'].append(csv_file)
                # meta_data['horizon'].append(int(df.shape[0] * 0.2))
                meta_data['horizon'].append(int(min(df.shape[0] * 0.2, 10 * frequency)))
                meta_data['frequency'].append(frequency)
                meta_data['nan_count'].append(int(df.isna().sum().values.sum()))
                meta_data['num_rows'].append(df.shape[0])
                meta_data['num_cols'].append(df.shape[1])
                meta_data['origin_index'].append(int(max(df.shape[0] * 0.4, (2 * frequency) + 1)))
                meta_data['step_size'].append(math.ceil((0.8 * df.shape[0]) / 100))

        metadata_df = pd.DataFrame(meta_data)
        metadata_df.to_csv(os.path.join(data_dir, '0_metadata.csv'), index=False)

        logger.info('Univariate forecasting data ready.')

    def format_global_forecasting_data(self, data_dir: str, gather_metadata: bool = False):
        """Prepare forecasting data for modelling from zip files

        :param str data_dir: Path to data directory
        :param bool gather_metadata: Store datasets metadata in a CSV file, defaults to False
        """
        tsf_files = self.extract_zip_tsf_files(data_dir)

        # Parse .tsf files sequentially
        meta_data: dict[str, list] = {
            'file': [],
            'frequency': [],
            'horizon': [],
            'has_nans': [],
            'equal_length': [],
            'num_rows': [],
            'num_cols': [],
        }
        for tsf_file in tsf_files:
            csv_path = os.path.join(data_dir, f'{tsf_file.split(".")[0]}.csv')

            # Parse .tsf files and output dataframe
            if not os.path.exists(csv_path) or gather_metadata:
                meta_data = self.convert_tsf_to_csv(data_dir, tsf_file, csv_path, meta_data)

        # Save dataset-specific metadata
        if gather_metadata:
            metadata_df = pd.DataFrame(meta_data)
            metadata_df.to_csv(os.path.join(data_dir, '0_metadata.csv'), index=False)

        logger.info('Global forecasting data ready.')

    def convert_tsf_to_csv(
        self,
        data_dir: str,
        tsf_file: str,
        csv_path: str,
        meta_data: dict,
    ) -> dict:
        """Convert .tsf file to CSV

        :param str data_dir: Path to data directory
        :param str tsf_file: Path to .tsf file
        :param str csv_path: Path to output CSV file
        :param dict meta_data: Metadata dictionary
        :return dict: Updated metadata dictionary
        """
        logger.info(f'Parsing {tsf_file}')
        # TODO: consider replacing this code with one of:
        # - sktime.datasets.load_from_tsfile
        # - aeon.datasets.load_from_tsfile
        data, freq, horizon, has_nans, equal_length = convert_tsf_to_dataframe(
            os.path.join(data_dir, tsf_file), 'NaN', 'value'
        )
        # Determine frequency
        if freq is None:
            raise NotImplementedError()
        else:
            freq = FREQUENCY_MAP[freq]
        # Determine horizon
        if horizon is None:
            horizon = self.select_horizon(freq, csv_path)
        else:
            freq = '1Y'

        # Parse data one variable at time
        df = pd.DataFrame()
        columns = []
        for row_index in range(len(data)):
            # Convert TSF row to CSV column
            column = self.process_row(data, row_index, freq)
            columns.append(column)
            if row_index % 1000 == 0:
                df = pd.concat([df] + columns, axis=1)
                columns = []

        # Concatenate remaining columns and save to CSV
        if len(columns) > 0:
            df = pd.concat([df] + columns, axis=1)
        df.to_csv(csv_path)

        # Update metadata
        meta_data['file'].append(tsf_file)
        meta_data['horizon'].append(horizon)
        meta_data['frequency'].append(freq)
        meta_data['has_nans'].append(has_nans)
        meta_data['equal_length'].append(equal_length)
        meta_data['num_rows'].append(df.shape[0])
        meta_data['num_cols'].append(df.shape[1])
        return meta_data

    def select_horizon(self, freq: str, csv_path: str | Path) -> int:
        """Select horizon for forecasters for a given dataset

        :param  str freq: Time series frequency
        :param str | Path csv_path: Path to CSV file
        :raises ValueError: If freq is None or not supported
        :return int: Forecasting horizon
        """
        # The following horizons are suggested by Godahewa et al. (2021)
        horizons: dict[str, int] = {
            'monthly': 12,
            'weekly': 8,
            'daily': 30,
            'hourly': 168,  # i.e. one week in hours
            'half_hourly': 168 * 2,  # i.e. one week in half-hours
            'minutely': 60 * 168,  # i.e. one week in minutes
        }

        if '4_seconds' in str(csv_path):
            horizon = 15  # i.e. 1 minute
        elif '10_minutes' in str(csv_path):
            horizon = 6  # i.e. 1 hour
        elif 'solar_weekly_dataset' in str(csv_path):
            horizon = 5
        elif freq is None:
            raise ValueError('No frequency or horizon found in file')
        else:
            try:
                horizon = horizons[freq]
            except KeyError as e:
                raise KeyError(f'Unclear what horizon to assign for frequency {freq}') from e
        return horizon

    def process_row(self, data, row_index, freq):
        """Convert Dataframe row to column with correct timestamp as index

        :param data: Original dataframe
        :param row_index: Index of row to process
        :param freq: Frequency of data
        :raises ValueError: Raised if dates exceed bounds processable by pandas
        :return: Pandas series
        """

        series = data.iloc[row_index, :]
        # Find series name, values and starting timestamp
        series_name = series.loc['series_name']
        values = series.loc['value']
        if 'start_timestamp' in data.columns:
            start_timestamp = series.loc['start_timestamp']
        else:
            start_timestamp = DatasetFormatter.default_start_timestamp

        # Format and apply date range index
        column = pd.DataFrame({series_name: values})
        for i in range(len(column)):
            try:
                # Create a datetime index
                timestamps = pd.date_range(
                    start=start_timestamp, periods=len(column) - i, freq=freq
                )

                # Truncating if too far into future
                if i > 0:
                    # Truncate by one extra period
                    periods = len(column) - (i + 1)
                    timestamps = pd.date_range(start=start_timestamp, periods=periods, freq=freq)
                    logger.warning(f'Truncating {series_name} from {len(column)} to {periods}')
                    column = column.head(periods)

                column = column.set_index(timestamps)
                break
            except pd.errors.OutOfBoundsDatetime as e:
                if i == len(column) - 1:
                    logger.error(series_name, start_timestamp, len(column), freq)
                    raise ValueError('Dates too far into the future for pandas to process') from e

        # Aggregate rows where timestamp index is apart by 1 second in hourly data
        if freq == '1H':
            column = column.resample('1H').mean()

        return column

    def extract_zip_tsf_files(self, data_dir: str) -> list[str]:
        """Read zip files from directory and extract .tsf files

        :param data_dir: Path to data directory of zip files
        :raises NotADirectoryError: occurs if non-directory passed as parameter
        :raises IOError: occurs if no zip files found
        :return: list of paths (str) to extracted .tsf files
        """
        # Validate input directory path
        try:
            zip_files = [f for f in os.listdir(data_dir) if f.endswith('zip')]
        except NotADirectoryError as e:
            raise NotADirectoryError(
                '\nProvide a path to a directory of zip files (of forecasting data)'
            ) from e

        if len(zip_files) == 0:
            raise IOError(f'\nNo zip files found in "{data_dir}"')

        # Extract zip files
        for filename in zip_files:
            with zipfile.ZipFile(Path(data_dir, filename)) as zip_file:
                # Check zip file contains exactly one .tsf file
                files = zip_file.namelist()
                if len(files) != 1 or not files[0].endswith('tsf'):
                    raise AssertionError('Zip files expected to contain exactly one .tsf file')
                # Extract zip file if not already extracted
                output_file = Path(data_dir, files[0])
                if not output_file.exists():
                    zip_file.extractall(data_dir)

        # Return list of extracted .tsf files
        tsf_files = [f for f in os.listdir(data_dir) if f.endswith('tsf')]
        return tsf_files
