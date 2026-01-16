"""AutoML via Python"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

DEFAULT_REPOSITORIES_PATH = Path('repositories')


@dataclass
class Library:
    """Dataclass for AutoML library information."""

    git_name: str
    git_url: str
    package_name: str

    def __str__(self):
        return self.package_name


AUTO_PYTORCH = Library(
    git_name='Auto-PyTorch',
    git_url='https://github.com/automl/Auto-PyTorch',
    package_name='autoPyTorch',
)
AUTO_SKLEARN = Library(
    git_name='auto-sklearn',
    git_url='https://github.com/automl/auto-sklearn',
    package_name='autosklearn',
)
AUTOGLUON = Library(
    git_name='AutoGluon',
    git_url='https://github.com/awslabs/autogluon',
    package_name='autogluon',
)
AUTO_KERAS = Library(
    git_name='AutoKeras',
    git_url='https://github.com/keras-team/autokeras',
    package_name='autokeras',
)
AUTO_TS = Library(
    git_name='AutoTS',
    git_url='https://github.com/winedarksea/AutoTS',
    package_name='autots',
)
EVAL_ML = Library(
    git_name='EvalML',
    git_url='https://github.com/alteryx/evalml',
    package_name='evalml',
)
FEDOT = Library(
    git_name='FEDOT',
    git_url='https://github.com/nccr-itmo/FEDOT',
    package_name='fedot',
)
# Fedot.Industrial
FEDOT_INDUSTRIAL = Library(
    git_name='Fedot.Industrial',
    git_url='https://github.com/aimclub/Fedot.Industrial',
    package_name='fedot_ind',
)
FLAML = Library(
    git_name='FLAML',
    git_url='https://github.com/microsoft/FLAML',
    package_name='flaml',
)
GAMA = Library(
    git_name='GAMA',
    git_url='https://github.com/openml-labs/gama',
    package_name='gama',
)
H2O = Library(
    git_name='H2O-3',
    git_url='https://github.com/h2oai/h2o-3',
    package_name='h2o',
)
HYPEROPT_SKLEARN = Library(
    git_name='hyperopt-sklearn',
    git_url='https://github.com/hyperopt/hyperopt-sklearn',
    package_name='hpsklearn',
)
HyperTS = Library(
    git_name='HyperTS',
    git_url='https://github.com/DataCanvasIO/HyperTS',
    package_name='hyperts',
)
KATS = Library(
    git_name='Kats',
    git_url='https://github.com/facebookresearch/Kats',
    package_name='kats',
)
LIGHT_AUTO_ML = Library(
    git_name='LightAutoML',
    git_url='https://github.com/AILab-MLTools/LightAutoML',
    package_name='lightautoml',
)
LUDWIG = Library(
    git_name='Ludwig',
    git_url='https://github.com/ludwig-ai/ludwig',
    package_name='ludwig',
)
ML_BOX = Library(
    git_name='MLBox',
    git_url='https://github.com/AxeldeRomblay/MLBox',
    package_name='mlbox',
)
MLJAR = Library(
    git_name='MLJAR Supervised',
    git_url='https://github.com/mljar/mljar-supervised',
    package_name='mljar-supervised',
)
PYCARET = Library(
    git_name='PyCaret',
    git_url='https://github.com/pycaret/pycaret',
    package_name='pycaret',
)
TPOT = Library(
    git_name='TPOT',
    git_url='https://github.com/epistasislab/tpot',
    package_name='tpot',
)

all_libraries = {
    AUTO_PYTORCH.package_name: AUTO_PYTORCH,
    AUTO_SKLEARN.package_name: AUTO_SKLEARN,
    AUTOGLUON.package_name: AUTOGLUON,
    AUTO_KERAS.package_name: AUTO_KERAS,
    AUTO_TS.package_name: AUTO_TS,
    EVAL_ML.package_name: EVAL_ML,
    FEDOT.package_name: FEDOT,
    FEDOT_INDUSTRIAL.package_name: FEDOT_INDUSTRIAL,
    FLAML.package_name: FLAML,
    GAMA.package_name: GAMA,
    H2O.package_name: H2O,
    HYPEROPT_SKLEARN.package_name: HYPEROPT_SKLEARN,
    KATS.package_name: KATS,
    LIGHT_AUTO_ML.package_name: LIGHT_AUTO_ML,
    LUDWIG.package_name: LUDWIG,
    ML_BOX.package_name: ML_BOX,
    MLJAR.package_name: MLJAR,
    PYCARET.package_name: PYCARET,
    TPOT.package_name: TPOT,
}
package_names = {lib.git_name: lib.package_name for lib in all_libraries.values()}
# Other libraries considered but not included:
# "https://github.com/tensorflow/adanet"    # Not AutoML
# "https://github.com/tinkoff-ai/etna"      # ETNA has been archived by owner
# "https://github.com/daochenzha/Meta-AAD"  # No updates
# "https://github.com/yzhao062/MetaOD"      # No updates
# "https://github.com/datamllab/pyodds"     # No updates since 2019


class BaseEnum(Enum):
    """Base Enum class with utility methods."""

    @classmethod
    def list(cls):
        """List all enum values."""
        return list(map(lambda c: c.value, cls))


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
