"""Module defining abstract and concrete machine learning tasks."""

from abc import ABC, abstractmethod
from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)

from ml import Library, TaskName
from ml.automl import AutoMLEngine
from ml.configuration import Configuration
from ml.dataset import (
    ClassificationDataset,
    DataFormatter,
    Dataset,
    DatasetReader,
    ForecastUnivariateDataset,
)


class BaseTask(ABC):
    """Abstract base class for different machine learning tasks."""

    config: Configuration
    dataset: Dataset

    def __init__(self, config: Configuration):
        self.config = config

    def read_data(self):
        """Read the data for the task."""
        self.dataset = DatasetReader(self.config).init_dataset()
        logger.debug('Dataset ready')

    def prepare_data(self):
        """Prepare/preprocess the data for the task."""
        self.dataset = DataFormatter(self.config).preprocess(self.dataset)
        logger.debug('Dataset preprocessed. Validating...')
        self._validate_dataset()

    def _validate_dataset(self):
        """Validate the dataset before training."""
        # Ensure dataset is initialized
        if self.dataset is None or self.dataset.df is None:
            raise ValueError('Dataset is not initialized or empty')
        # Check classification dataset has labels
        if isinstance(self.dataset, ClassificationDataset) and len(self.dataset.labels) == 0:
            raise ValueError(f'Classification dataset has no labels: {self.dataset.labels}')
        # Check training/val/test/etc. splits exist
        for split in ['train', 'val', 'train_val', 'test']:
            X = getattr(self.dataset, f'X_{split}', None)
            y = getattr(self.dataset, f'y_{split}', None)
            if X is None or y is None:
                continue
            if len(X) != len(y):
                raise ValueError(f'Mismatched lengths: X_{split}={len(X)} and y_{split}={len(y)}')
        # Ensure dataset has target column(s)
        if not self.dataset.target_cols:
            raise ValueError('Dataset missing target_cols')
        # Ensure only single target column for TS classification
        if self.config.task == TaskName.CLASSIFICATION and len(self.dataset.target_cols) != 1:
            raise NotImplementedError('TS classification only implemented for single target column')

    def call_libraries(self):
        """Call the necessary AutoML libraries for the task."""
        if len(self.config.libraries) == 0:
            logger.warning('No libraries specified. Skipping training...')
            return
        # Call each library
        for library in self.config.libraries:
            # Check if a Dockerfile exists for the library
            dockerfile_path = Path('src', 'ml', 'automl', library, 'Dockerfile')
            if not dockerfile_path.exists():
                logger.error(f'Dockerfile not found for library: {library}. Skipping...')
                continue
            # Check if models.py exists for the library
            python_path = Path('src', 'ml', 'automl', library, 'models.py')
            if not python_path.exists():
                logger.error(f'models.py not found for library: {library}. Skipping...')
                continue

            logger.info(f'Running {library} for task: {self.config.task}')
            _ = self._call_library(library)

    def _call_library(self, library: str) -> pd.Series | None:
        """Call a specific AutoML library for the task.

        :param str library: Name of library
        :raises ValueError: Raised for unknown library name
        :return pd.Series | None: Predictions or None if library is skipped
        """
        engine = self._init_engine(library)
        # Skip if the library is not implemented for the task
        if engine is None:
            logger.warning(f'No engine initialized for library: {library}. Skipping...')
            predictions = None
            return {}

        logger.info(f'Training {library} for task: {self.config.task}')
        # Separate features and target variables
        y = self.dataset.df[self.dataset.target_cols]
        X = self.dataset.df.drop(columns=self.dataset.target_cols)

        # Initialise cross-validator
        validator, val_kwargs, n_splits = self._cross_validator()
        # Perform cross-validation
        fold = 0
        all_scores = {}
        val_used_folds = []
        for train_index, test_index in validator.split(X, y, **val_kwargs):
            fold += 1
            logger.info(f'Fold {fold}')
            # Get training and testing data for this fold
            X_train_val, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train_val, y_test = y.iloc[train_index], y.iloc[test_index]

            # CV by group
            if isinstance(validator, StratifiedGroupKFold):
                # Training folds
                train_folds = val_kwargs['groups'].iloc[train_index].unique().tolist()
                # Specify fold for validation
                val_fold_id = [f for f in train_folds if f not in val_used_folds][0]
                val_used_folds.append(val_fold_id)  # Ensure each fold used once for validation
                # Testing fold
                test_fold_id = val_kwargs['groups'].iloc[test_index].unique()[0]

                logger.debug(f'Validation Group: {val_fold_id}')
                logger.debug(f'Test Set Group: {test_fold_id}')
                fold_name = 'fold-' + test_fold_id

                # Split data
                X_train = X_train_val[val_kwargs['groups'] != val_fold_id]
                y_train = y_train_val[val_kwargs['groups'] != val_fold_id]
                X_val = X_train_val[val_kwargs['groups'] == val_fold_id]
                y_val = y_train_val[val_kwargs['groups'] == val_fold_id]
            # CV (K=5)
            elif isinstance(validator, StratifiedKFold):
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_val,
                    y_train_val,
                    test_size=1 / n_splits,
                    stratify=y_train_val,
                    random_state=self.config.random_state,
                )
                fold_name = fold
            else:
                raise NotImplementedError()

            # Train models via the specified AutoML library
            # X_train, X_val, X_test = self._filter_columns(X_train, X_val, X_test)
            engine.train(X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val)
            # Make predictions
            logger.info(f'Training {library} finished. Predicting for task: {self.config.task}')
            predictions = engine.predict(X_test)
            all_scores[f'fold-{fold_name}'] = self._evaluate_results(predictions, y_test)
        return all_scores

    @abstractmethod
    def _init_engine(self, library: str) -> AutoMLEngine | None:
        """Dynamically import the wrapper for the specified library.

        :param str library: Name of library
        :return AutoMLEngine | None: Instance of the AutoML engine or None if not implemented
        """

    def _cross_validator(self, **kwargs) -> tuple[BaseCrossValidator, dict, int]:
        """Create a cross validation object.

        :raises ValueError: Raised for unknown validation method
        :return tuple: Cross-validator, additional kwargs, number of splits
        """
        validator_kwargs = {}
        # Shuffled K-Fold cross-validation
        if self.config.validation.startswith('cv-'):
            n_splits = int(self.config.validation.split('-')[-1])
        # K-Fold cross-validation by group
        elif self.config.validation.startswith('group-'):
            if 'group_col' not in kwargs:
                raise ValueError('Missing "group_col" argument for grouped CV')
            groups, n_splits = self._split_by_group(self.dataset.df, kwargs['group_col'])
            validator_kwargs['groups'] = groups
        else:
            raise ValueError(f'Unknown validation method: {self.config.validation}')

        split_kwargs = {
            'n_splits': n_splits,
            'random_state': self.config.random_state,
            'shuffle': True,
        }
        # K-Fold cross-validation by shuffling
        if self.config.validation.startswith('cv-'):
            if isinstance(self.dataset, ClassificationDataset):
                validator = StratifiedKFold(**split_kwargs)
            elif isinstance(self.dataset, ForecastUnivariateDataset):
                del split_kwargs['shuffle']
                validator = TimeSeriesSplit(**split_kwargs)
                # Defaults: max_train_size=None, test_size=None, gap=0
            else:
                raise NotImplementedError(f'K-Fold CV not implemented for: {type(self.dataset)}')
        # K-Fold cross-validation by group
        elif self.config.validation.startswith('group-'):
            if isinstance(self.dataset, ClassificationDataset):
                validator = StratifiedGroupKFold(**split_kwargs)
            else:
                raise NotImplementedError(f'Grouped CV not implemented for: {type(self.dataset)}')

        return validator, validator_kwargs, n_splits

    def _split_by_group(self, df: pd.DataFrame, group_col: str):
        """Split DataFrame by group based on the specified 'group_col' column.

        :param pd.DataFrame df: DataFrame to split
        :param str group_col: Name of group column
        :raises ValueError: Raised if group_col is missing or insufficient groups found
        :return tuple[pd.Series, int]: Series of group values and number of unique groups
        """
        if group_col not in df.columns:
            raise ValueError(f'DataFrame missing "{group_col}" column')
        # Ensure multiple groups found
        num_groups = df[group_col].nunique()
        if num_groups < 2:
            raise ValueError(f'Found only {num_groups} unique groups. Need at least 2.')
        return df[group_col].tolist(), num_groups

    @abstractmethod
    def _evaluate_results(self, predictions: pd.Series, y_true: pd.DataFrame | pd.Series):
        """Evaluate the results of the task."""

    def run(self):
        """Run the task."""
        self.read_data()
        self.prepare_data()
        self.call_libraries()


class PrepareDataTask(BaseTask):
    """Task to download and preprocess data without modelling."""

    def call_libraries(self):
        """No modelling."""

    def _evaluate_results(self, _, __):
        """No modelling."""

    def _init_engine(self, __) -> AutoMLEngine | None:
        """No modelling."""
        return None


class TSClassificationTask(BaseTask):
    """Time series classification task."""

    def call_libraries(self):
        """Call the necessary AutoML libraries for the task."""
        if TaskName(self.config.task) != TaskName.CLASSIFICATION:
            raise ValueError(f'Invalid task for classifier: {self.config.task}')
        super().call_libraries()

    def _evaluate_results(self, predictions: pd.Series, y_true: pd.Series):
        """Evaluate the results of the task."""
        if len(self.config.libraries) == 0:
            logger.warning('No libraries specified. Skipping evaluation...')
            return
        raise NotImplementedError

    def _init_engine(self, library: str) -> AutoMLEngine | None:
        """Dynamically import the classifier class for the specified library.

        :param str library: Name of library
        :raises ValueError: Raised for unknown library name
        :return AutoMLEngine | None: Instance of the AutoML engine or None if not implemented
        """
        # pylint: disable=C0415:import-outside-toplevel
        # fmt: off
        if library == Library.EvalML:
            from ml.automl.evalml.models import EvalMLEngine
            engine = EvalMLEngine(config=self.config)
        if library == Library.FLAML:
            from ml.automl.flaml.models import FLAMLEngine
            engine = FLAMLEngine(config=self.config)
        elif library in Library.list():
            engine = None
            logger.warning(f'Library {library} not implemented for task {self.config.task}')
        else:
            raise ValueError(f'Unknown library: {library}')
        # fmt: on
        return engine


class SentimentTSClassificationTask(TSClassificationTask):

    def _cross_validator(self, **kwargs) -> tuple[BaseCrossValidator, dict, int]:
        """Create a cross validation object. Add group_id column for grouped CV."""
        # Add 'session' or 'speaker' column for grouped CV
        if self.config.validation.startswith('group-'):
            group_col = self.config.validation.split('-')[-1]  # e.g., 'session', 'speaker'
            if group_col not in self.dataset.df.columns:
                raise ValueError(f'Dataset missing "{group_col}" column for grouped CV')
            groups = self.dataset.df[group_col]
            logger.debug(f'Audio groups: \n{groups.value_counts()}')
        return super()._cross_validator(group_col=group_col, **kwargs)


class TaskBuilder:
    """Factory class to create task instances based on the task name."""

    @staticmethod
    def init_task(config: Configuration) -> BaseTask:
        """Get the task instance based on the task name.

        :param Configuration config: Configuration object.
        :return AbstractTask: Instance of the task.
        """
        logger.info(f'Initializing time series task: "{TaskName.CLASSIFICATION}"')
        if config.task == TaskName.CLASSIFICATION:
            return TSClassificationTask(config)
        if config.task == TaskName.PREPARE_DATA:
            return PrepareDataTask(config)
        raise ValueError(f'Unknown task name: "{config.task}"')
