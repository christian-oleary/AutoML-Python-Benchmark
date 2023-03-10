"""
Utility functions
"""

import os
import logging
import sys

logger_name = 'Benchmark'
logger = logging.getLogger(logger_name)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(level=logging.DEBUG)

class Utils:
    """Utility functions"""

    logger = logging.getLogger(logger_name)

    @staticmethod
    def get_csv_datasets(datasets_directory):
        """Fetch list of file names of CSV datasets

        :param datasets_directory: Path to datasets directory (str)
        :raises NotADirectoryError: If datasets_directory does not exist
        :raises IOError: If datasets_directory does not have CSV files
        :return: list of dataset file names
        """

        if not os.path.exists(datasets_directory):
            raise NotADirectoryError('Datasets direcotry path does not exist')

        csv_files = [
            f for f in os.listdir(datasets_directory)
            if f.endswith('csv') and '0_metadata' not in f
            ]

        if len(csv_files) == 0:
            raise IOError('No CSV files found')

        return csv_files
