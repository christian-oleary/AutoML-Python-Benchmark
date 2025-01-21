"""Credit to Rakshitha Godahewa et al.

Adapted from: https://github.com/rakshitha123/TSForecasting/blob/master/utils/data_loader.py
"""

# cspell: ignore file
# flake8: noqa
# pylint: skip-file

from datetime import datetime
from distutils.util import strtobool

import pandas as pd

# Map from: https://github.com/rakshitha123/TSForecasting/
FREQUENCY_MAP = {
    '4_seconds': '4S',
    'minutely': '1min',
    '10_minutes': '10min',
    '5_minutes': '5min',
    'half_hourly': '30min',
    'hourly': '1H',
    'daily': '1D',
    'weekly': '1W',
    'monthly': '1M',
    'quarterly': '1Q',
    'yearly': '1Y',
}

REVERSE_FREQUENCY_MAP = {v: k for k, v in FREQUENCY_MAP.items()}


# Converts the contents in a .tsf file into a dataframe and returns it along with other meta-data of the dataset: frequency, horizon, whether the dataset contains missing values and whether the series have equal lengths
#
# Parameters
# full_file_path_and_name - complete .tsf file path
# replace_missing_vals_with - a term to indicate the missing values in series in the returning dataframe
# value_column_name - Any name that is preferred to have as the name of the column containing series values in the returning dataframe
def convert_tsf_to_dataframe(
    full_file_path_and_name, replace_missing_vals_with='NaN', value_column_name='series_value'
):
    """Converts .tsf file into a dataframe"""
    try:
        with open(full_file_path_and_name, 'r', encoding='utf-8') as file:
            return parse_file(file, replace_missing_vals_with, value_column_name)
    except UnicodeDecodeError:
        with open(full_file_path_and_name, 'r', encoding='cp1252') as file:
            return parse_file(file, replace_missing_vals_with, value_column_name)


def parse_file(file, replace_missing_vals_with, value_column_name):
    """From Rakshitha Godahewa et al."""
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    for line in file:
        # Strip white space from start/end of line
        line = line.strip()

        if line:
            if line.startswith('@'):  # Read meta-data
                if not line.startswith('@data'):
                    line_content = line.split(' ')
                    if line.startswith('@attribute'):
                        if len(line_content) != 3:  # Attributes have both name and type
                            raise Exception('Invalid meta-data specification.')

                        col_names.append(line_content[1])
                        col_types.append(line_content[2])
                    else:
                        if len(line_content) != 2:  # Other meta-data have only values
                            raise Exception('Invalid meta-data specification.')

                        if line.startswith('@frequency'):
                            frequency = line_content[1]
                        elif line.startswith('@horizon'):
                            forecast_horizon = int(line_content[1])
                        elif line.startswith('@missing'):
                            contain_missing_values = bool(strtobool(line_content[1]))
                        elif line.startswith('@equallength'):
                            contain_equal_length = bool(strtobool(line_content[1]))

                else:
                    if len(col_names) == 0:
                        raise Exception(
                            'Missing attribute section. Attribute section must come before data.'
                        )

                    found_data_tag = True
            elif not line.startswith('#'):
                if len(col_names) == 0:
                    raise Exception(
                        'Missing attribute section. Attribute section must come before data.'
                    )
                elif not found_data_tag:
                    raise Exception('Missing @data tag.')
                else:
                    if not started_reading_data_section:
                        started_reading_data_section = True
                        found_data_section = True
                        all_series = []

                        for col in col_names:
                            all_data[col] = []

                    full_info = line.split(':')

                    if len(full_info) != (len(col_names) + 1):
                        raise Exception('Missing attributes/values in series.')

                    series = full_info[len(full_info) - 1]
                    series = series.split(',')

                    if len(series) == 0:
                        raise Exception(
                            'A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol'
                        )

                    numeric_series = []

                    for val in series:
                        if val == '?':
                            numeric_series.append(replace_missing_vals_with)
                        else:
                            numeric_series.append(float(val))

                    if numeric_series.count(replace_missing_vals_with) == len(numeric_series):
                        raise Exception(
                            'All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series.'
                        )

                    all_series.append(pd.Series(numeric_series).array)

                    for i in range(len(col_names)):
                        att_val = None
                        if col_types[i] == 'numeric':
                            att_val = int(full_info[i])
                        elif col_types[i] == 'string':
                            att_val = str(full_info[i])
                        elif col_types[i] == 'date':
                            att_val = datetime.strptime(full_info[i], '%Y-%m-%d %H-%M-%S')
                        else:
                            raise Exception(
                                'Invalid attribute type.'
                            )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                        if att_val is None:
                            raise Exception('Invalid attribute value.')
                        else:
                            all_data[col_names[i]].append(att_val)

            line_count = line_count + 1

    if line_count == 0:
        raise Exception('Empty file.')
    if len(col_names) == 0:
        raise Exception('Missing attribute section.')
    if not found_data_section:
        raise Exception('Missing series information under data section.')

    all_data[value_column_name] = all_series
    loaded_data = pd.DataFrame(all_data)

    return (
        loaded_data,
        frequency,
        forecast_horizon,
        contain_missing_values,
        contain_equal_length,
    )


# Example of usage
# loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe('TSForecasting/tsf_data/sample.tsf')

# print(loaded_data)
# print(frequency)
# print(forecast_horizon)
# print(contain_missing_values)
# print(contain_equal_length)
