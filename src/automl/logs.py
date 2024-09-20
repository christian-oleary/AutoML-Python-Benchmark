"""Logging"""

from enum import Enum
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys


class BaseEnum(Enum):
    """Base class for enums"""

    def __str__(self):
        """Return string representation of enum"""
        return self.value.lower()

    @classmethod
    def _missing_(cls, value):
        """Return enum member from string value

        :param str value: string value of enum
        :return Enum: enum member
        """
        value = value.lower()
        for member in cls:
            if member.__str__().lower() == value:  # pylint: disable=unnecessary-dunder-call
                return member
        return None

    @classmethod
    def get_options(cls):
        """Get list of enum values"""
        return [e.__str__().lower() for e in list(cls)]  # pylint: disable=unnecessary-dunder-call


class LogLevel(BaseEnum):
    """Log levels"""

    CRITICAL = 'CRITICAL'
    ERROR = 'ERROR'
    WARNING = 'WARNING'
    INFO = 'INFO'
    DEBUG = 'DEBUG'


# https://docs.python.org/3/library/logging.html#logrecord-attributes
# log_format = '|%(module)s.py|%(funcName)s|line %(lineno)d|%(asctime)s|%(levelname)s|: %(message)s'
# log_format = '|%(module)s.py|%(funcName)s|%(asctime)s|%(levelname)s|: %(message)s'
log_format = '|%(asctime)s|%(levelname)s|: %(message)s'

time_format = '%Y-%m-%d %H:%M:%S'


class ColorFormatter(logging.Formatter):
    """Format logs to use colors for log levels"""

    grey = "\x1b[30;20m"
    white = "\x1b[37;20m"
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: cyan + log_format + reset,
        logging.INFO: green + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record) -> str:
        """Format log record.

        :param record: log record
        :return str: formatted log record
        """
        formatter = logging.Formatter(self.FORMATS.get(record.levelno), time_format)
        return formatter.format(record)


logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').disabled = True

# Set up logger to print to console
logger = logging.getLogger(__name__)


def set_log_dir(log_dir='logs'):
    """Format logs for log files"""
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            os.path.join(log_dir, 'log'), when='H', backupCount=24, utc=True
        )
        file_handler.setFormatter(logging.Formatter(log_format, time_format))
        logger.addHandler(file_handler)
        logger.info(f'Logging to directory: {log_dir}')


# Format logs for stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(ColorFormatter())
logger.addHandler(stream_handler)

# Set log level
logger.setLevel(level=logging.DEBUG)
logger.propagate = False

# Reduce logs from matplotlib
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').disabled = True
