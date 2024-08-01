"""
Definitions of custom errors
"""


class DatasetTooSmallError(Exception):
    """Should be called when a dataset is too small for a library to handle"""

    def __init__(self, message: str, *args: object) -> None:
        super().__init__(message, *args)


class AutomlLibraryError(Exception):
    """Should be called when an AutoML library fails to fit or predict"""

    def __init__(self, message, *args: object):
        super().__init__(message, *args)
