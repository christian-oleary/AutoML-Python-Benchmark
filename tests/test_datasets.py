"""Unit tests for the ml.datasets.py file"""

# pylint: disable=redefined-outer-name,unused-argument

from pathlib import Path

import pandas as pd
import pytest

from ml.dataset import Dataset, ISEMDataset, RAVDESS

DATASET_SIZE = 10


class DummyDataset(Dataset):
    """Dummy implementation for testing Dataset abstract class."""

    def _init_dataset(self, **kwargs):
        self.df = pd.DataFrame({'a': list(range(DATASET_SIZE)), 'b': list(range(DATASET_SIZE))})
        return self


class DummyUtils:
    """Mock Utils for find_files_by_extension."""

    @staticmethod
    def find_files_by_extension(data_dir, ext, recursive, absolute):
        """Mock implementation that generates file paths based on expected RAVDESS format."""
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
        """Dataset subclass with empty DataFrame for testing ensure_data."""

        def _init_dataset(self, **kwargs):
            self.data = None
            self.df = pd.DataFrame()
            return self

    with pytest.raises(ValueError):
        EmptyDataset(name='dummy')


@pytest.fixture
def sample_csv(tmp_path):
    """Fixture to create a sample CSV file for testing ISEMDataset."""
    # Create a sample CSV file with 'applicable_date' column
    data = {'applicable_date': ['2020-01-01', '2020-01-02', '2020-01-03'], 'value': [100, 200, 300]}
    df = pd.DataFrame(data)
    csv_path = tmp_path / 'isem.csv'
    df.to_csv(csv_path, index=False)
    return csv_path


def test_isem_dataset_init_reads_csv(sample_csv):
    """Test that ISEMDataset initializes correctly and reads the CSV file."""
    ds = ISEMDataset(path=sample_csv)
    assert isinstance(ds.df, pd.DataFrame)
    assert 'value' in ds.df.columns
    # Index should be 'applicable_date'
    assert ds.df.index.name == 'applicable_date'
    assert len(ds.df) == 3


def test_isem_dataset_missing_path_raises():
    """Test that initializing ISEMDataset without a path raises ValueError."""
    with pytest.raises(ValueError, match='No path provided for I-SEM dataset'):
        ISEMDataset(path=None)


def test_isem_dataset_missing_applicable_date_column(tmp_path):
    """Create CSV without 'applicable_date'."""
    csv_path = tmp_path / 'bad.csv'
    pd.DataFrame({'date': ['2020-01-01'], 'value': [100]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match='Missing applicable_date column'):
        ISEMDataset(path=csv_path)


def test_isem_dataset_ensure_data(sample_csv):
    """Test that ensure_data raises ValueError if df is empty."""
    ds = ISEMDataset(path=sample_csv)
    ds.ensure_data()  # Should not raise
    ds.df = pd.DataFrame()  # Empty DataFrame
    with pytest.raises(ValueError, match='Empty dataset!'):
        ds.ensure_data()
