import os
import zipfile

import pandas as pd

from src.data_loader import convert_tsf_to_dataframe


class Utils:

    # Maps from: https://github.com/rakshitha123/TSForecasting/
    FREQUENCY_MAP = {
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

    @staticmethod
    def format_forecasting_data(data_dir, debug=False):

        tsf_files = Utils.extract_forecasting_data(data_dir, debug)

        if debug: # Testing occurs with one file
            tsf_files = [tsf_files[0]]

        for tsf_file in tsf_files:
            csv_path = os.path.join(data_dir, f'{tsf_file.split(".")[0]}.csv')

            if not os.path.exists(csv_path):
                print(f'Parsing {tsf_file}')
                data, freq, horizon, has_nans, equal_length = convert_tsf_to_dataframe(
                    os.path.join(data_dir, tsf_file), 'NaN', 'value')

                if freq is not None:
                    freq = Utils.FREQUENCY_MAP[freq]
                else:
                    freq = '1Y'

                df = pd.DataFrame()
                for row in range(len(data)):
                    series = data.iloc[row, :]
                    series_name = series.loc['series_name']
                    start_timestamp = series.loc['start_timestamp']
                    values = series.loc['value']

                    column = pd.DataFrame({series_name: values})
                    timestamps = pd.date_range(start=start_timestamp, periods=len(column), freq=freq)
                    column = column.set_index(timestamps)

                    df = pd.concat([df, column], axis=1)
                    df.to_csv(csv_path)
            else:
                print(csv_path, pd.read_csv(csv_path).shape, 'already exists. Skipping...')


    def extract_forecasting_data(data_dir, debug=False):
        """Read zip files from directory and

        :param data_dir: Path to data directory of zip files
        :param debug: _description_, defaults to False
        :raises NotADirectoryError: _description_
        :raises IOError: _description_
        :return: _description_
        """
        # Validate input directory path
        try:
            zip_files = [ f for f in os.listdir(data_dir) if f.endswith('zip') ]
        except NotADirectoryError as e:
            raise NotADirectoryError('\nProvide a path to a directory of zip files (of forecasting data)')

        if len(zip_files) == 0:
            raise IOError(f'\nNo zip files found in "{data_dir}"')

        if debug: # Just test with one zip file
            zip_files = [zip_files[0]]

        # Extract zip files
        for filename in zip_files:
            zip_file = zipfile.ZipFile(os.path.join(data_dir, filename))
            files = zip_file.namelist()
            error_msg = 'Zip files expected to contain exactly one .tsf file'
            assert len(files) == 1 and files[0].endswith('tsf'), error_msg
            output_file = os.path.join(data_dir, files[0])
            if not os.path.exists(output_file) and not debug:
                zip_file.extractall(data_dir)

        tsf_files = [ f for f in os.listdir(data_dir) if f.endswith('tsf') ]
        return tsf_files
