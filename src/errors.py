"""
Definitions of sutom errors
"""

class DatasetTooSmallError(Exception):
    """Should be called when a dataset is too small for a library to handle"""

    def __init__(self, message, errors):
        super(DatasetTooSmallError, self).__init__(message)
