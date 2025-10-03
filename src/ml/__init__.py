"""AutoML via Python"""

from enum import Enum


# Library names and their display names
class Library(str, Enum):  # noqa: cohesion
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


class TaskName(str, Enum):
    """Modelling task names."""

    ANOMALY_DETECTION = 'anomaly_detection'
    CLASSIFICATION = 'classification'
    FORECASTING = 'forecasting'
    NONE = 'none'
    PREPARE_DATA = 'prepare_data'


# Labels of interest for audio classification tasks
AUDIO_LABELS = ['anger', 'happy', 'sad', 'neutral']
