"""
Definitions of sutom errors
"""

class DatasetTooSmallError(Exception):
    """Should be called when a dataset is too small for a library to handle"""

    def __init__(self, message, errors=None):
        super(DatasetTooSmallError, self).__init__(message)


class AutomlLibraryError(Exception):
    """Should be called when an AutoML library fails to fit or predict"""

    def __init__(self, message, errors):
        super(AutomlLibraryError, self).__init__(message)
