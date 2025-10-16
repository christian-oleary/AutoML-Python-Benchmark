"""AutoML via Python"""

from enum import Enum


class BaseEnum(Enum):
    """Base Enum class with utility methods."""

    @classmethod
    def list(cls):
        """List all enum values."""
        return list(map(lambda c: c.value, cls))


# Library names and their display names
class Library(str, BaseEnum):  # noqa: cohesion
    """AutoML Libraries."""

    AutoGluon = 'autogluon'
    AutoKeras = 'autokeras'
    AutoTS = 'autots'
    AutoPyTorch = 'auto_pytorch'
    AutoSklearn = 'auto_sklearn'
    # ETNA = 'etna'  # ETNA has been archived by owner
    EvalML = 'evalml'
    FEDOT = 'fedot'
    FLAML = 'flaml'
    # GAMA = 'gama'
    # H2O = 'h2o_3'
    HyperoptSklearn = 'hyperopt_sklearn'
    # KATS = 'kats'
    LightAutoML = 'lightautoml'
    Ludwig = 'ludwig'
    MLBox = 'mlbox'
    MLJAR = 'mljar_supervised'
    TPOT = 'tpot'
    PyCaret = 'pycaret'


class TaskName(str, BaseEnum):
    """Modelling task names."""

    ANOMALY_DETECTION = 'anomaly_detection'
    CLASSIFICATION = 'classification'
    FORECAST_UNIVARIATE = 'forecast_univariate'
    FORECAST_MULTIVARIATE = 'forecasting_multivariate'
    NONE = 'none'
    PREPARE_DATA = 'prepare_data'


# Labels of interest for audio classification tasks
AUDIO_LABELS = ['anger', 'happy', 'sad', 'neutral']


class AudioColumns(BaseEnum):
    """Columns for speech emotion recognition (SER) via time series classification (TSC)"""

    PATH = 'path'
    AUDIO = 'audio'
    LABEL = 'label'
    TEXT = 'text'
