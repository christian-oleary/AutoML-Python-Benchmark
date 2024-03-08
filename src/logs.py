import os
from datetime import datetime
import logging
from logging.handlers import TimedRotatingFileHandler
import sys

# https://docs.python.org/3/library/logging.html#logrecord-attributes
# log_format = '|%(module)s.py|%(funcName)s|line %(lineno)d|%(asctime)s|%(levelname)s|: %(message)s'
# log_format = '|%(module)s.py|%(funcName)s|%(asctime)s|%(levelname)s|: %(message)s'
log_format = '|%(asctime)s|%(levelname)s|: %(message)s'

time_format = '%Y-%m-%d %H:%M:%S'

class ColourFormatter(logging.Formatter):
    """Format logs to use colours for log levels"""

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
        logging.CRITICAL: bold_red + log_format + reset
    }

    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format, time_format)
        return formatter.format(record)


logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').disabled = True

# Set up logger to print to console
logger = logging.getLogger('Benchmark')

# Format logs for log files
def set_log_dir(log_dir='logs'):
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = TimedRotatingFileHandler(os.path.join(log_dir, 'log'),
                                                when='H', backupCount=24, utc=True)
        file_handler.setFormatter(logging.Formatter(log_format, time_format))
        logger.addHandler(file_handler)
        logger.info(f'Logging to directory: {log_dir}')

# Format logs for stdout
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(ColourFormatter())
logger.addHandler(stream_handler)

# Set log level
logger.setLevel(level=logging.DEBUG)

# Reduce logs from matplotlib
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').disabled = True
