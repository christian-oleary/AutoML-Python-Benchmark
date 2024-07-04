"""
Data formatting functions
"""

import argparse
from datetime import datetime
import logging
import os
import math
import zipfile

import pandas as pd

from src.frequencies import frequencies
from src.logs import logger
from src.TSForecasting.data_loader import convert_tsf_to_dataframe, FREQUENCY_MAP
from src.validation import Task


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

    def format_univariate_forecasting_data(self, data_dir):
        """Format data for univariate forecasting

        :param str data_dir: Data directory
        """

        headers_and_timestamps = (
            'libra' not in data_dir
        )  # Libra dataset is missing indices and headers

        logger.info('Reading univariate forecasting data...')

        meta_data = {
            'file': [],
            'horizon': [],
            'frequency': [],
            'nan_count': [],
            'num_rows': [],
            'num_cols': [],
        }

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
                meta_data['nan_count'].append(int(df.isna().sum()))
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
                meta_data['nan_count'].append(int(df.isna().sum()))
                meta_data['num_rows'].append(df.shape[0])
                meta_data['num_cols'].append(df.shape[1])
                meta_data['origin_index'].append(int(max(df.shape[0] * 0.4, (2 * frequency) + 1)))
                meta_data['step_size'].append(math.ceil((0.8 * df.shape[0]) / 100))

        metadata_df = pd.DataFrame(meta_data)
        metadata_df.to_csv(os.path.join(data_dir, '0_metadata.csv'), index=False)

        logger.info('Univariate forecasting data ready.')

    def format_global_forecasting_data(self, data_dir, gather_metadata=False):
        """Prepare forecasting data for modelling from zip files

        :param str data_dir: Path to data directory
        :param bool gather_metadata: Store datasets metadata in a CSV file, defaults to False
        """

        tsf_files = self.extract_forecasting_data(data_dir)

        if gather_metadata:
            meta_data = {
                'file': [],
                'frequency': [],
                'horizon': [],
                'has_nans': [],
                'equal_length': [],
                'num_rows': [],
                'num_cols': [],
            }

        # Parse .tsf files sequentially
        for tsf_file in tsf_files:
            csv_path = os.path.join(data_dir, f'{tsf_file.split(".")[0]}.csv')

            # Parse .tsf files and output dataframe
            if not os.path.exists(csv_path) or gather_metadata:
                logger.info(f'Parsing {tsf_file}')
                data, freq, horizon, has_nans, equal_length = convert_tsf_to_dataframe(
                    os.path.join(data_dir, tsf_file), 'NaN', 'value'
                )

                if horizon is None:
                    horizon = self.select_horizon(freq, csv_path)

                if gather_metadata:
                    meta_data['file'].append(tsf_file)
                    meta_data['horizon'].append(horizon)
                    meta_data['frequency'].append(freq)
                    meta_data['has_nans'].append(has_nans)
                    meta_data['equal_length'].append(equal_length)

                # if not os.path.exists(csv_path):
                # Determine frequency
                if freq is not None:
                    freq = FREQUENCY_MAP[freq]
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
                if len(columns) > 0:
                    df = pd.concat([df] + columns, axis=1)

                df.to_csv(csv_path)
                if gather_metadata:
                    meta_data['num_rows'].append(df.shape[0])
                    meta_data['num_cols'].append(df.shape[1])

        # Save dataset-specific metadata
        if gather_metadata:
            metadata_df = pd.DataFrame(meta_data)
            metadata_df.to_csv(os.path.join(data_dir, '0_metadata.csv'), index=False)

        logger.info('Global forecasting data ready.')

    def select_horizon(self, freq, csv_path):
        """Select horizon for forecasters for a given dataset

        :param freq: Time series frequency (str)
        :param csv_path: Path to dataset (str)
        :raises ValueError: If freq is None or not supported
        :return: Forecasting horizon (int)
        """
        if '4_seconds' in csv_path:
            horizon = 15  # i.e. 1 minute
        elif '10_minutes' in csv_path:
            horizon = 6  # i.e. 1 hour

        # The following horizons are suggested by Godahewa et al. (2021)
        elif 'solar_weekly_dataset' in csv_path:
            horizon = 5
        elif freq is None:
            raise ValueError('No frequency or horizon found in file')
        elif freq == 'monthly':
            horizon = 12
        elif freq == 'weekly':
            horizon = 8
        elif freq == 'daily':
            horizon = 30
        elif freq == 'hourly':
            horizon = 168  # i.e. one week
        elif freq == 'half_hourly':
            horizon = 168 * 2  # i.e. one week
        elif freq == 'minutely':
            horizon = 60 * 168  # i.e. one week
        else:
            raise ValueError(f'Unclear what horizon to assign for frequency {freq}')
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
                    timestamps = pd.date_range(
                        start=start_timestamp, periods=len(column) - (i + 1), freq=freq
                    )
                    logging.warning(
                        f'Truncating {series_name} from {len(column)} to {len(column)-(i+1)}'
                    )
                    column = column.head(len(column) - (i + 1))

                column = column.set_index(timestamps)
                break
            except pd.errors.OutOfBoundsDatetime as e:
                if i == len(column) - 1:
                    logging.error(series_name, start_timestamp, len(column), freq)
                    raise ValueError('Dates too far into the future for pandas to process') from e

        # Aggregate rows where timestamp index is apart by 1 second in hourly data
        if freq == '1H':
            column = column.resample('1H').mean()

        return column

    def extract_forecasting_data(self, data_dir):
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
            with zipfile.ZipFile(os.path.join(data_dir, filename)) as zip_file:
                files = zip_file.namelist()
                error_msg = 'Zip files expected to contain exactly one .tsf file'
                assert len(files) == 1 and files[0].endswith('tsf'), error_msg
                output_file = os.path.join(data_dir, files[0])
                if not os.path.exists(output_file):
                    zip_file.extractall(data_dir)

        tsf_files = [f for f in os.listdir(data_dir) if f.endswith('tsf')]
        return tsf_files

    def format_anomaly_data(self, data_dir):
        """Format anomaly detection datasets

        :param data_dir: Path to directory of datasets
        """

        self.format_3W_data(data_dir)
        self.format_falling_data(data_dir)
        self.format_beth_data(data_dir)
        self.format_hai_data(data_dir)
        self.format_nab_data(data_dir)
        self.format_skab_data(data_dir)

    def format_3W_data(self, data_dir):
        """Format 3W data

        :param data_dir: Path to directory of datasets
        """
        # subdir = os.path.join(data_dir, '3W')

    def format_falling_data(self, data_dir):
        """Format falling data

        :param data_dir: Path to directory of datasets
        """

    def format_beth_data(self, data_dir):
        """Format 3W data

        :param data_dir: Path to directory of datasets
        """

    def format_hai_data(self, data_dir):
        """Format HAI Security Dataset data

        :param data_dir: Path to directory of datasets
        """

    def format_nab_data(self, data_dir):
        """Format Numenta Anomaly detection Benchmark data

        :param data_dir: Path to directory of datasets
        """

    def format_skab_data(self, data_dir):
        """Format Skoltech Anomaly Benchmark data

        :param data_dir: Path to directory of datasets
        """
