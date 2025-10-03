"""Unit tests for the ml.datasets.py file"""

from pathlib import Path

import pandas as pd
import pytest

from ml.dataset import Dataset, ISEMDataset, RAVDESS

DATASET_SIZE = 10


class DummyDataset(Dataset):
    """Dummy implementation for testing Dataset abstract class."""

    def _init_dataset(self, **kwargs):
        # Create a simple DataFrame for testing
        self.df = pd.DataFrame({'a': list(range(DATASET_SIZE)), 'b': list(range(DATASET_SIZE))})
        return self


class DummyUtils:
    """Mock Utils for find_files_by_extension."""

    @staticmethod
    def find_files_by_extension(data_dir, ext, recursive, absolute):
        # Generate dummy file paths with correct stem format
        # Format: '03-01-01-01-01-01-01.wav' (emotion=01, statement=01)
        stems = [f'03-01-0{(i%8)+1}-01-0{(i%2)+1}-01-01' for i in range(RAVDESS.expected_rows)]
        return [Path(f'/dummy/path/{stem}.wav') for stem in stems]


def test_dataset_missing_name():
    """Test Dataset initialization with missing name raises ValueError."""
    with pytest.raises(ValueError):
        DummyDataset(name=None)


def test_dataset_file_not_found(tmp_path):
    """Test Dataset initialization with non-existent file raises FileNotFoundError."""
    missing_file = tmp_path / 'notfound.csv'
    with pytest.raises(FileNotFoundError):
        DummyDataset(name=missing_file)


def test_ensure_data_empty():
    """Test ensure_data raises ValueError if df is empty."""

    class EmptyDataset(Dataset):
        def _init_dataset(self, **kwargs):
            self.df = pd.DataFrame()
            return self

    with pytest.raises(ValueError):
        EmptyDataset(name='dummy')


def test_split_data_splits_correctly():
    """Test split_data splits data into train and test sets."""
    ds = DummyDataset(name='dummy')
    train, test = ds.split_data()
    assert len(train) == 8
    assert len(test) == 2
    assert train.equals(ds.train_df)
    assert test.equals(ds.test_df)


@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
    """Patch Utils.find_files_by_extension in RAVDESS._init_dataset"""
    monkeypatch.setattr(
        'ml.dataset.Utils.find_files_by_extension', DummyUtils.find_files_by_extension
    )


def test_ravdess_init_creates_dataframe(tmp_path, monkeypatch):
    # Patch _download_from_huggingface to do nothing
    monkeypatch.setattr(
        'ml.dataset.RAVDESS._download_from_huggingface', lambda self, n_jobs=1: None
    )
    ds = RAVDESS(data_dir=tmp_path)
    assert isinstance(ds.df, pd.DataFrame)
    assert len(ds.df) == RAVDESS.expected_rows
    assert set(ds.df.columns) >= {'path', 'label', 'text'}


def test_ravdess_wrong_row_count_raises(monkeypatch):
    # Patch _download_from_huggingface to do nothing
    monkeypatch.setattr(
        'ml.dataset.RAVDESS._download_from_huggingface', lambda self, n_jobs=1: None
    )
    # Patch find_files_by_extension to return fewer files
    monkeypatch.setattr(
        'ml.dataset.Utils.find_files_by_extension',
        lambda *a, **k: [Path(f'/dummy/03-01-0{i}-01-01-01-01.wav') for i in range(1, 9)],
    )
    with pytest.raises(ValueError, match="Wrong number of rows"):
        RAVDESS(data_dir="dummy")


def test_ravdess_unused_labels_excludes(monkeypatch):
    monkeypatch.setattr(  # Patch _download_from_huggingface to do nothing
        'ml.dataset.RAVDESS._download_from_huggingface', lambda self, n_jobs=1: None
    )

    # Patch find_files_by_extension to return files with all emotions
    def files_with_all_emotions(*a, **k):
        stems = [f'03-01-0{e}-01-01-01-01' for e in range(1, 9)]
        return [Path(f'/dummy/{stem}.wav') for stem in stems]

    monkeypatch.setattr('ml.dataset.Utils.find_files_by_extension', files_with_all_emotions)
    RAVDESS.expected_rows = 6
    ds = RAVDESS(data_dir='dummy', unused_labels=['neutral', 'happy'])
    assert 'neutral' not in ds.df['label'].values
    assert 'happy' not in ds.df['label'].values


def test_ravdess_split_data(monkeypatch, tmp_path):
    monkeypatch.setattr(
        'ml.dataset.RAVDESS._download_from_huggingface', lambda self, n_jobs=1: None
    )
    ds = RAVDESS(data_dir=tmp_path)
    train, test = ds.split_data()
    assert len(train) == int(RAVDESS.expected_rows * 0.8)
    assert len(test) == int(RAVDESS.expected_rows * 0.2)
    assert train.equals(ds.train_df)
    assert test.equals(ds.test_df)


@pytest.fixture
def sample_csv(tmp_path):
    # Create a sample CSV file with 'applicable_date' column
    data = {'applicable_date': ['2020-01-01', '2020-01-02', '2020-01-03'], 'value': [100, 200, 300]}
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'isem.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def test_isem_dataset_init_reads_csv(sample_csv):
    ds = ISEMDataset(path=sample_csv)
    assert isinstance(ds.df, pd.DataFrame)
    assert 'value' in ds.df.columns
    # Index should be 'applicable_date'
    assert ds.df.index.name == 'applicable_date'
    assert len(ds.df) == 3


def test_isem_dataset_missing_path_raises():
    with pytest.raises(ValueError, match='No path provided for I-SEM dataset'):
        ISEMDataset(path=None)


def test_isem_dataset_missing_applicable_date_column(tmp_path):
    """Create CSV without 'applicable_date'."""
    csv_path = tmp_path / 'bad.csv'
    pd.DataFrame({'date': ['2020-01-01'], 'value': [100]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match='Missing applicable_date column'):
        ISEMDataset(path=csv_path)


def test_isem_dataset_ensure_data(sample_csv):
    ds = ISEMDataset(path=sample_csv)
    ds.ensure_data()  # Should not raise
    ds.df = pd.DataFrame()  # Empty DataFrame
    with pytest.raises(ValueError, match='Empty dataset!'):
        ds.ensure_data()


def test_isem_dataset_split_data(sample_csv):
    ds = ISEMDataset(path=sample_csv)
    train, test = ds.split_data()
    # 80% train, 20% test (with 3 rows: 2 train, 0 test due to int truncation)
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) == 2
    assert len(test) == 0
    assert train.equals(ds.train_df)
    assert test.equals(ds.test_df)
