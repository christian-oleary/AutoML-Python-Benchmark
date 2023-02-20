"""
Miscellaneous utility functions
"""

from datetime import datetime
import os
import logging
import zipfile

import pandas as pd

from src.data_loader import convert_tsf_to_dataframe


class Utils:
    """General utility functions"""

    # Maps from: https://github.com/rakshitha123/TSForecasting/
    FREQUENCY_MAP = {
        '4_seconds': '4S',
        'minutely': '1min',
        '10_minutes': '10min',
        'half_hourly': '30min',
        'hourly': '1H',
        'daily': '1D',
        'weekly': '1W',
        'monthly': '1M',
        'quarterly': '1Q',
        'yearly': '1Y'
    }

    default_start_timestamp = datetime.strptime('1970-01-01 00-00-00', '%Y-%m-%d %H-%M-%S')

    logger = logging.getLogger('Benchmark')

    @staticmethod
    def format_forecasting_data(data_dir, debug=False):
        """Prepare forecasting data for modelling from zip files"""

        tsf_files = Utils.extract_forecasting_data(data_dir, debug)

        if debug: # Testing occurs with one file
            tsf_files = [tsf_files[0]]
        # tsf_files = sorted(tsf_files, reverse=True)

        # Parse .tsf files sequentially
        meta_data = { 'file': [], 'frequency': [], 'horizon': [], 'has_nans': [], 'equal_length': [], 'num_cols': [] }
        for tsf_file in tsf_files:
            csv_path = os.path.join(data_dir, f'{tsf_file.split(".")[0]}.csv')
            meta_data['file'].append(tsf_file)

            if not os.path.exists(csv_path):
                Utils.logger.info(f'Parsing {tsf_file}')
                # Parse .tsf files and output dataframe
                data, freq, horizon, has_nans, equal_length = convert_tsf_to_dataframe(
                    os.path.join(data_dir, tsf_file), 'NaN', 'value')
                meta_data['horizon'].append(horizon)
                meta_data['frequency'].append(freq)
                meta_data['has_nans'].append(has_nans)
                meta_data['equal_length'].append(equal_length)
                meta_data['num_cols'].append(len(data))

                # Determine frequency
                if freq is not None:
                    freq = Utils.FREQUENCY_MAP[freq]
                else:
                    freq = '1Y'

                # Parse data one variable at time
                Utils.format_to_csv(data, freq, csv_path)
            else:
                Utils.logger.info(f'{csv_path} {pd.read_csv(csv_path).shape} already exists. Skipping...')

        # Save dataset-specific metadata
        metadata_df = pd.DataFrame(meta_data)
        metadata_df.to_csv(os.path.join(data_dir, '0_metadata.csv'), index=False)


    @staticmethod
    def extract_forecasting_data(data_dir, debug=False):
        """Read zip files from directory and

        :param data_dir: Path to data directory of zip files
        :param debug: If true, only use one file for testing, defaults to False
        :raises NotADirectoryError: occurs if non-directory passed as parameter
        :raises IOError: occurs if no zip files found
        :return: list of paths (str) to extracted .tsf files
        """
        # Validate input directory path
        try:
            zip_files = [ f for f in os.listdir(data_dir) if f.endswith('zip') ]
        except NotADirectoryError as e:
            raise NotADirectoryError('\nProvide a path to a directory of zip files (of forecasting data)') from e

        if len(zip_files) == 0:
            raise IOError(f'\nNo zip files found in "{data_dir}"')

        if debug: # Just test with one zip file
            zip_files = [zip_files[0]]

        # Extract zip files
        for filename in zip_files:
            with zipfile.ZipFile(os.path.join(data_dir, filename)) as zip_file:
                files = zip_file.namelist()
                error_msg = 'Zip files expected to contain exactly one .tsf file'
                assert len(files) == 1 and files[0].endswith('tsf'), error_msg
                output_file = os.path.join(data_dir, files[0])
                if not os.path.exists(output_file) and not debug:
                    zip_file.extractall(data_dir)

        tsf_files = [ f for f in os.listdir(data_dir) if f.endswith('tsf') ]
        return tsf_files


    @staticmethod
    def format_to_csv(data, freq, csv_path):
        """Extract data from .tsf to .csv tabular format

        :param data: Data read from .tsf file
        :param freq: Frequency of data (str)
        :param csv_path: Path to output CSV file (str)
        :raises ValueError: occurs if timestamps extend too far into future for pandas to process (year >= )
        """
        df = pd.DataFrame()

        for row in range(len(data)):
            series = data.iloc[row, :]
            # Find series name, values and starting timestamp
            series_name = series.loc['series_name']
            values = series.loc['value']
            if 'start_timestamp' in data.columns:
                start_timestamp = series.loc['start_timestamp']
            else:
                start_timestamp = Utils.default_start_timestamp

            # Format and apply date range index
            column = pd.DataFrame({series_name: values})
            for i in range(len(column)):
                try:
                    timestamps = pd.date_range(start=start_timestamp, periods=len(column)-i, freq=freq)

                    if i > 0: # Truncating if too far into future
                        # Truncate by one extra period
                        timestamps = pd.date_range(start=start_timestamp, periods=len(column)-(i+1), freq=freq)
                        Utils.logger.warn(f'Truncating {series_name} from {len(column)} to {len(column)-(i+1)}')
                        column = column.head(len(column)-(i+1))
                    column = column.set_index(timestamps)
                    break
                except pd._libs.tslibs.np_datetime.OutOfBoundsDatetime as e:
                    if i == len(column)-1:
                        Utils.logger.error(series_name, start_timestamp, len(column), freq)
                        raise ValueError(f'Pandas cannot handle timestamps beyond {pd.Timestamp.max}') from e

            column.to_csv('column.csv')

            # Join to main dataframe
            df = pd.concat([df, column], axis=1)
        df.to_csv(csv_path)


    @staticmethod
    def format_anomaly_data(data_dir, debug=False):
        """Format anomaly detection datasets

        :param data_dir: Path to directory of datasets
        :param debug: Used during testing, defaults to False
        """

        Utils.format_3W_data(data_dir, debug)
        Utils.format_falling_data(data_dir, debug)
        Utils.format_BETH_data(data_dir, debug)
        Utils.format_HAI_data(data_dir, debug)
        Utils.format_NAB_data(data_dir, debug)
        Utils.format_SKAB_data(data_dir, debug)


    @staticmethod
    def format_3W_data(data_dir, debug=False):
        """Format 3W data

        :param data_dir: Path to directory of datasets
        :param debug: Used during testing, defaults to False
        """
        subdir = os.path.join(data_dir, '3W')


    @staticmethod
    def format_falling_data(data_dir, debug=False):
        """Format falling data

        :param data_dir: Path to directory of datasets
        :param debug: Used during testing, defaults to False
        """


    @staticmethod
    def format_BETH_data(data_dir, debug=False):
        """Format 3W data

        :param data_dir: Path to directory of datasets
        :param debug: Used during testing, defaults to False
        """


    @staticmethod
    def format_HAI_data(data_dir, debug=False):
        """Format HAI Security Dataset data

        :param data_dir: Path to directory of datasets
        :param debug: Used during testing, defaults to False
        """


    @staticmethod
    def format_NAB_data(data_dir, debug=False):
        """Format Numenta Anomaly detection Benchmark data

        :param data_dir: Path to directory of datasets
        :param debug: Used during testing, defaults to False
        """


    @staticmethod
    def format_SKAB_data(data_dir, debug=False):
        """Format Skoltech Anomaly Benchmark data

        :param data_dir: Path to directory of datasets
        :param debug: Used during testing, defaults to False
        """
