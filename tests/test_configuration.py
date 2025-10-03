"""Unit tests for the ml.configuration file"""

import pytest

from ml import Library
from ml.configuration import Configuration, LogLevelEnum


def test_default_configuration():
    config = Configuration()
    assert config.cpu_only is True
    assert config.dataset is None
    assert str(config.data_dir) == 'data'
    assert config.libraries == []
    assert config.log_file is None
    assert config.log_level == LogLevelEnum.DEBUG
    assert config.n_jobs == 1
    assert str(config.output_dir) == 'results'
    assert str(config.preprocessed_subdir) == 'preprocessed'
    assert config.random_state == 1
    assert config.task is None
    assert config.verbosity == 1


def test_configuration_with_all_libraries(monkeypatch):
    config = Configuration(libraries=['all'])
    all_libs = [lib.value for lib in Library]
    assert sorted(config.libraries) == sorted(all_libs)


def test_configuration_with_none_libraries():
    config = Configuration(libraries=['none'])
    assert config.libraries == []


def test_configuration_with_prepare_data_libraries():
    config = Configuration(libraries=['prepare_data'])
    assert config.libraries == []


def test_configuration_with_invalid_library():
    with pytest.raises(ValueError):
        Configuration(libraries=['invalid_lib'])


def test_configuration_with_valid_libraries():
    libs = [lib.value for lib in Library][:2]
    config = Configuration(libraries=libs)
    assert sorted(config.libraries) == sorted(libs)


def test_env_prefix(monkeypatch):
    monkeypatch.setenv('N_JOBS', '5')
    config = Configuration()
    assert config.n_jobs == 5


def test_log_file_set(tmp_path):
    log_path = tmp_path / 'log.txt'
    config = Configuration(log_file=log_path)
    assert str(config.log_file) == str(log_path)
