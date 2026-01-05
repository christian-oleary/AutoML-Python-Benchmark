"""Logging configuration."""

import logging
import sys

from loguru import logger

logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


def add_relative_path(record):
    """WIP: Add relative path to loguru records. Currently incomplete."""
    record['extra']['relative_path'] = record['name'].replace('.', '/') + '.py'


class Logs:
    """Logging setup."""

    DEFAULT_FORMAT = ''.join(
        '|<green>{time:YYYY-MM-DD HH:mm:ss}</green>|<level>{level: <8}</level>|'
        '<cyan>{file.path}</cyan>:<cyan>{line: <3}</cyan>| <level>{message}</level>'
        # '<cyan>{extra[relative_path]}</cyan>:<cyan>{line: <3}</cyan>| <level>{message}</level>'
    )

    STARTED = False

    @classmethod
    def log_to_stderr(
        cls,
        log_format=DEFAULT_FORMAT,
        level='DEBUG',
        colorize=True,
        backtrace=False,
        diagnose=False,
        enqueue=True,
    ):
        """Set up logging to stderr."""
        if not cls.STARTED:
            logger.remove()
            cls.STARTED = True

        logger.add(
            sys.stderr,
            format=log_format,
            level=level,
            colorize=colorize,
            backtrace=backtrace,
            diagnose=diagnose,
            enqueue=enqueue,
        )
        logger.configure(patcher=add_relative_path)

    @classmethod
    def log_to_file(
        cls,
        log_format=DEFAULT_FORMAT,
        sink='logs/ml.log',
        level='DEBUG',
        backtrace=True,
        diagnose=True,
        rotation='30 MB',
        retention='14 days',
        compression='zip',
        enqueue=True,
    ):
        """Set up logging to file."""
        if not cls.STARTED:
            logger.remove()
            cls.STARTED = True

        logger.add(
            sink=sink,
            format=log_format,
            level=level,
            backtrace=backtrace,
            diagnose=diagnose,
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=enqueue,
        )
        logger.configure(patcher=add_relative_path)
