"""Data formatting functions."""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import datasets
import pandas as pd
from pandas import DataFrame
from typing_extensions import Self

# from ml.frequencies import frequencies
from ml import AUDIO_LABELS, AudioColumns, TaskName
from ml.configuration import Configuration
from ml.logs import logger
from ml.utils import Utils


class Dataset(ABC):
    """Base class for datasets."""

    # Default location to store datasets
    data_dir: str | Path = Path('data')
    preprocessed_dir: str | Path = Path('data', 'preprocessed')
    df_path: str | Path = Path('data', 'preprocessed', 'df.csv')

    # Dataset aliases, name and Hugging Face Hub alias
    aliases: list[str] = []
    name: str | Path | None = None
    hub_name: str | None = None

    # Dataframes
    df: pd.DataFrame
    target_cols: list[str] | None = None

    # Task Type
    task: TaskName = TaskName.NONE

    def __init__(self, **kwargs):
        """Initialize dataset.
        :param str | Path name: Name of dataset or path to dataset file
        :param str | Path data_dir: Path to the data directory
        :param bool init_dataset: Whether to initialize the dataset (download/read data)
        """
        self.name = kwargs.get('name', self.name)
        self.data_dir = kwargs.get('data_dir', self.data_dir)
        self.target_cols = kwargs.get('target_cols', self.target_cols)

        # Validate name
        if not self.name:
            raise ValueError('No name provided for dataset')

        if isinstance(self.name, Path) or (
            isinstance(self.name, str) and self.name.endswith('csv')
        ):
            if not Path(self.name).exists():
                raise FileNotFoundError(f'Path not found: {self.name}')

        # Fetch data
        if kwargs.get('init_dataset', True):
            self._init_dataset(**kwargs)
            self.ensure_data()

    @abstractmethod
    def _init_dataset(self, **kwargs) -> Self:
        """Fetch data relating to dataset."""

    def ensure_data(self):
        """Ensure dataset is not empty.

        :raises ValueError: If dataset is empty
        """
        if self.df is None or len(self.df) == 0:
            raise ValueError('Empty dataset!')

    def _download_from_huggingface(self, n_jobs: int = 1) -> tuple[Path, datasets.DatasetDict]:
        """Download dataset from Hugging Face Hub.

        :param int n_jobs: Number of processes to use for download, defaults to 1
        :return tuple[Path, datasets.DatasetDict]: Path to data directory and dataset object
        """
        output_dir = Path(self.data_dir, str(self.name))
        if output_dir.exists():
            self.dataset: datasets.DatasetDict = datasets.load_from_disk(str(output_dir))
            logger.debug(f'Using existing dataset at: {output_dir}')
        else:
            # Remove existing intermediate directory
            # cache_dir = Path(self.data_dir, f'{self.name}_DOWNLOAD_TEMP')
            # if cache_dir.exists():
            #     Utils.delete_paths(cache_dir)
            # cache_dir.mkdir(exist_ok=True, parents=True)
            # Start download
            logger.info(f'Downloading "{self.name}" from Hugging Face Hub at: "{self.hub_name}"')
            self.dataset = datasets.load_dataset(
                str(self.hub_name),
                num_proc=n_jobs,
                split=None,
                trust_remote_code=True,
                # data_dir=str(cache_dir), # cache_dir=str(cache_dir),
            )
            logger.info(f'Downloaded dataset. Saving to: {output_dir}')
            self.dataset.save_to_disk(output_dir, num_proc=n_jobs)
            self.dataset.cleanup_cache_files()

        logger.info(f'Dataset: {self.dataset} ({type(self.dataset)})')
        if len(list(output_dir.glob('*'))) == 0:
            raise FileNotFoundError('Downloaded dataset directory is empty!')
        # Utils.delete_paths(cache_dir)  # Delete temporary download directory
        return output_dir, self.dataset


class ClassificationDataset(Dataset):
    """Base class for classification datasets."""

    task: TaskName = TaskName.CLASSIFICATION
    labels: list[str] = []
    target_cols: list[str] = [AudioColumns.LABEL.value]

    @abstractmethod
    def _init_dataset(self, **kwargs) -> Self:
        """Fetch data relating to dataset."""
        return self


class AudioClassificationDataset(ClassificationDataset):
    """Audio classification dataset."""

    labels: list[str] = AUDIO_LABELS

    @abstractmethod
    def _init_dataset(self, **kwargs) -> Self:
        """Fetch data relating to dataset."""
        return self


class RAVDESS(AudioClassificationDataset):
    """Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset."""

    aliases: list[str] = ['ravdess']
    name = 'ravdess'
    hub_name = 'narad/ravdess'

    expected_rows: int = 1440
    unused_labels: list[str] = []

    def _init_dataset(self, **kwargs) -> Self:
        """Fetch RAVDESS data."""
        self.unused_labels = kwargs.get('unused_labels', self.unused_labels)
        self.target_cols = [AudioColumns.LABEL.value]
        # Download data
        self._download_from_huggingface(int(kwargs.get('n_jobs', 1)))
        # Read labels
        self.emotion_map = {
            '01': 'neutral',
            '02': 'calm',
            '03': 'happy',
            '04': 'sad',
            '05': 'anger',
            '06': 'fear',
            '07': 'disgust',
            '08': 'surprise',
        }
        statement_map = {'01': 'Kids are talking by the door', '02': 'Dogs are sitting by the door'}
        # modality_map = {'01': 'audio_and_video', '02': 'video_only', '03': 'audio_only'}
        # vocal_channel_map = {'01': 'speech', '02': 'song'}
        # intensity_map = {'01': 'normal', '02': 'strong'}
        # repetition_map = {'01': '1st_repetition', '02': '2nd_repetition'}
        data = []
        for file_path in Utils.find_files_by_extension(self.data_dir, 'wav', True, True):
            parts = file_path.stem.split('-')
            emotion = self.emotion_map[parts[2]]
            if emotion not in self.unused_labels:  # type: ignore
                entry = {
                    'path': file_path,
                    self.target_cols[0]: emotion,
                    'text': statement_map[parts[4]],
                    # 'modality': modality_map[parts[0]],
                    # 'vocal_channel': vocal_channel_map[parts[1]],
                    # 'intensity': intensity_map[parts[3]],
                    # 'repetition': repetition_map[parts[5]],
                    'speaker': parts[6],
                }
                data.append(entry)
        # Validate data
        self.df = pd.DataFrame(data)
        if len(self.df) != self.expected_rows:
            raise ValueError(f'Wrong row count: {len(self.df)}. Expected: {self.expected_rows}')
        return self

    # def handle_downloaded_data(self, **kwargs):
    #     """Handle downloaded data from Hugging Face Hub."""
    #     # Get name of download directory (dynamically generated name)
    #     data_dir = Path(kwargs['cache_dir'], 'downloads', 'extracted')
    #     data_dir = Path(str(list(data_dir.glob('*'))[0]).replace('.lock', ''))
    #     # Move dataset to expected location
    #     os.rename(data_dir, kwargs['output_dir'])


class IEMOCAP(AudioClassificationDataset):
    """Interactive Emotional Dyadic Motion Capture (IEMOCAP) dataset."""

    aliases: list[str] = ['iemocap']
    name = 'iemocap'
    hub_name = 'AbstractTTS/IEMOCAP'
    columns: dict = {
        AudioColumns.PATH.value: 'file',
        AudioColumns.AUDIO.value: 'audio',
        AudioColumns.LABEL.value: 'major_emotion',
        AudioColumns.TEXT.value: 'transcription',
    }
    # feature_cols = ['speaking_rate', 'pitch_mean', 'pitch_std', 'rms', 'relative_db']

    expected_rows: int = 10039
    unused_labels: list[str] = []

    def _init_dataset(self, **kwargs) -> Self:
        """Fetch IEMOCAP data."""
        self.unused_labels = kwargs.get('unused_labels', self.unused_labels)
        self.target_cols = [AudioColumns.LABEL.value]
        # Download (or load) data
        self._download_from_huggingface(int(kwargs.get('n_jobs', 1)))
        # Convert to pandas dataframe
        self._ds_to_pandas()  # .to_csv(Path('test.csv'), index=False)
        if len(self.df) != self.expected_rows:
            raise ValueError(f'Wrong row count: {len(self.df)}. Expected: {self.expected_rows}')
        return self

    def _ds_to_pandas(self) -> pd.DataFrame:
        """Convert a datasets.DatasetDict to a pandas DataFrame."""
        if not isinstance(self.dataset, datasets.DatasetDict):
            raise ValueError('Dataset is not a datasets.DatasetDict object')
        # Filter columns, add split name, and concatenate splits into a dataframe
        splits = [
            self.dataset[key].select_columns(self.columns.values()).to_pandas().assign(split=key)
            for key in self.dataset.keys()
        ]
        self.df = pd.concat(splits, ignore_index=True)
        # Rename columns to standard names
        self.df = self.df.rename(columns={v: k for k, v in self.columns.items()})
        return self.df


class ForecastUnivariateDataset(Dataset):
    """Base class for forecasting datasets."""

    task: TaskName = TaskName.FORECAST_UNIVARIATE

    # Forecasting horizon and frequency
    frequency: str
    horizon: int
    # Start and end times for training and test sets
    train_set_start_time: str | None = None
    train_set_end_time: str | None = None
    test_set_start_time: str | None = None
    test_set_end_time: str | None = None

    @abstractmethod
    def _init_dataset(self, **kwargs) -> Self:
        """Fetch data relating to dataset."""
        self.frequency = kwargs.get('frequency', self.frequency)
        self.horizon = int(kwargs.get('horizon', self.horizon))
        self.train_set_start_time = kwargs.get('train_set_start_time', self.train_set_start_time)
        self.train_set_end_time = kwargs.get('train_set_end_time', self.train_set_end_time)
        self.test_set_start_time = kwargs.get('test_set_start_time', self.test_set_start_time)
        self.test_set_end_time = kwargs.get('test_set_end_time', self.test_set_end_time)
        return self

    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into training and test sets."""
        self.ensure_data()
        # Holdout for model testing (80% training, 20% testing).
        # This seems to be used by Godahewa et al. for global forecasting:
        # https://github.com/rakshitha123/TSForecasting/blob/master/experiments/rolling_origin.R#L10
        # Also used by Bauer 2021 for univariate forecasting
        self.train_df = self.df.head(int(len(self.df) * 0.8))
        self.test_df = self.df.tail(int(len(self.df) * 0.2))
        return self.train_df, self.test_df


class ISEMDataset(Dataset):
    """I-SEM dataset."""

    aliases: list[str] = ['isem']
    name: str = 'isem'
    task: TaskName = TaskName.FORECAST_UNIVARIATE

    frequency: str = '24H'
    horizon: int = 24
    has_nans: bool = False

    def _init_dataset(self, **kwargs) -> Self:
        """Fetch I-SEM data.

        :raises ValueError: If no path provided
        """
        self.path = kwargs.get('path')
        if not self.path:
            raise ValueError('No path provided for I-SEM dataset')

        # Read I-SEM data. Expecting 'applicable_date' column to use as index
        self.df = pd.read_csv(self.path)
        if 'applicable_date' not in self.df.columns:
            raise ValueError('Missing applicable_date column')
        self.df = self.df.set_index('applicable_date')
        return self


class ISEM2020Dataset(ISEMDataset):
    """I-SEM 2020 dataset."""

    aliases: list[str] = ['isem2020']
    name: str = 'isem2020'
    train_set_start_time: str = '2019/12/31 23:00:00'
    train_set_end_time: str = '2020/10/19 23:00:00'
    test_set_start_time: str = '2020/10/20 00:00:00'
    test_set_end_time: str = '2020/12/31 22:00:00'

    def split_data(self) -> tuple[DataFrame, DataFrame]:
        """Split data into training and test sets."""
        self.ensure_data()
        logger.debug('Loading I-SEM 2020 dataset')
        # Ensure index is datetime
        self.df.index = pd.to_datetime(self.df.index)
        self.train_df = self.df.loc[self.train_set_start_time : self.train_set_end_time, :]  # type: ignore
        self.test_df = self.df.loc[self.test_set_start_time : self.test_set_end_time, :]  # type: ignore
        return self.train_df, self.test_df


class DatasetReader:
    """Methods for formatting raw datasets in preparation for modelling."""

    default_start_timestamp = datetime.strptime('1970-01-01 00-00-00', '%Y-%m-%d %H-%M-%S')

    def __init__(self, config: Configuration):
        self.config = config
        self.dataset = None

    def init_dataset(self):
        """Determine if data is a named dataset or a path."""
        if self.config.dataset in RAVDESS.aliases:
            self.dataset = RAVDESS(data_dir=self.config.data_dir)
        elif self.config.dataset in IEMOCAP.aliases:
            self.dataset = IEMOCAP(data_dir=self.config.data_dir)
        else:
            raise NotImplementedError(f'Dataset {self.config.dataset} is not supported')
        return self.dataset


class DataFormatter:
    """Preprocess datasets for modelling."""

    def __init__(self, config: Configuration):
        self.config = config

    def preprocess(self, dataset: Dataset) -> Dataset:
        """Preprocess a dataset for a specific task type.

        :param Dataset dataset: Dataset to be processed
        """
        if self.config.task in [TaskName.CLASSIFICATION, TaskName.PREPARE_DATA]:
            if isinstance(dataset, AudioClassificationDataset):
                dataset = self._preprocess_audio(dataset)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return dataset

    def _preprocess_audio(self, dataset: Dataset) -> Dataset:
        """Preprocess audio data for time series classification.

        :param Dataset dataset: Dataset to be processed
        :return Dataset: Preprocessed dataset
        """
        self.preprocessed_dir = Path(  # Specify dir for preprocessed data
            self.config.data_dir, self.config.preprocessed_subdir, str(dataset.name)
        )
        self.df_path = Path(self.preprocessed_dir, 'df.csv')

        dataset.preprocessed_dir = self.preprocessed_dir
        dataset.df_path = self.df_path

        # If preprocessed data already exists, return that
        # if self.df_path.exists():
        #     logger.info(f'Using existing dataframe at: {self.df_path}')
        #     dataset.df = pd.read_csv(self.df_path)
        #     return dataset

        # Drop unused labels and check data shapes
        if isinstance(dataset, AudioClassificationDataset):
            # Drop unused labels
            logger.debug(f'{dataset.name} original shape: {dataset.df.shape}')
            original_length = len(dataset.df)
            dataset.df = dataset.df[dataset.df[AudioColumns.LABEL.value].isin(dataset.labels)]
            # Check new dataframe length
            new_length = original_length - len(dataset.df)
            logger.debug(f'Removed {new_length} rows. Shape: {dataset.df.shape}')
            if len(dataset.df) == original_length:
                logger.warning('All labels are being used for classification')

        # Ensure df has data
        if len(dataset.df) == 0:
            raise ValueError('No data in dataframe "df"')
        # Ensure df has multiple labels if classification
        if isinstance(dataset, ClassificationDataset) and len(dataset.labels) < 2:
            raise ValueError('At least two labels are required for classification')

        # Save preprocessed data
        # self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        # dataset.df.to_csv(self.df_path, index=False)
        # logger.info(f'Saved dataframe "df" to: {self.df_path}')
        return dataset


# class LibraDataset(ForecastingDataset):
#     self.data = metadata[metadata['file'] == csv_file]
#     self.frequency = int(self.data['frequency'].iloc[0])
#     self.horizon = int(self.data['horizon'].iloc[0])
#     self.actual = self.test_df.copy().values.flatten()
#     self.y_train = self.train_df.copy().values.flatten()  # Required for MASE
#     # Libra's custom rolling origin forecast:
#     kwargs = {
#         'origin_index': int(self.data['origin_index'].iloc[0]),
#         'step_size': int(self.data['step_size'].iloc[0])
#         }
#     self.df = pd.read_csv(self.dataset_path, header=None)
#     self.df.columns = ['target']


# class MonashDataset(ForecastingDataset):
#     # Filter datasets based on "Monash Time Series Forecasting Archive" by Godahewa et al. (2021)
#     # we do not consider the London smart meters, wind farms, solar power, and wind power datasets
#     # for both univariate and global model evaluations, the Kaggle web traffic daily dataset for
#     # the global model evaluations and the solar 10 minutely dataset for the WaveNet evaluation
#     filter_forecast_datasets = True  # To do:  make an env variable
#     if filter_forecast_datasets and self.dataset_name in self.omitted_datasets:
#         logger.debug(f'Skipping dataset {self.dataset_name}')
#         continue

#     # self.data = metadata[metadata['file'] == csv_file.replace('csv', 'tsf')]
#     # self.frequency = self.data['frequency'].iloc[0]
#     # self.horizon = self.data['horizon'].iloc[0]
#     # if pd.isna(self.horizon):
#     #     raise ValueError(f'Missing horizon in 0_metadata.csv for {csv_file}')
#     # self.horizon = int(self.horizon)
#     # # To do: revise frequencies, determine and data formatting stage
#     # if pd.isna(self.frequency) and 'm3_other_dataset.csv' in csv_file:
#     #     self.frequency = 'yearly'
#     # self.actual = self.test_df.values

#     self.df = pd.read_csv(self.dataset_path, index_col=0)
