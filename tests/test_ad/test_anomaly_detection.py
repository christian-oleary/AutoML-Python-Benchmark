"""Unit tests for anomaly_detection.py."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ml.ad.anomaly_detection import BaseADModel, PyCaretADModel

# pylint: disable=redefined-outer-name


@pytest.fixture
def sample_df():
    """A small DataFrame with numeric features and an anomaly label column."""
    return pd.DataFrame(
        {
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'anomaly': [0, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def features_df():
    """A DataFrame without a label column (for predict / predict_proba)."""
    return pd.DataFrame({'feature1': [1.0, 2.0, 3.0], 'feature2': [10.0, 20.0, 30.0]})


# ---------------------------------------------------------------------------
# BaseADModel
# ---------------------------------------------------------------------------
class ConcreteADModel(BaseADModel):
    """Minimal concrete subclass used to test BaseADModel helpers."""

    def __init__(self):
        self.fitted_df = None

    def _fit(self, df: pd.DataFrame, **kwargs) -> None:
        """Store the DataFrame for testing purposes."""
        self.fitted_df = df

    def predict(self, df: pd.DataFrame):
        """Return a dummy Series for testing purposes."""
        return pd.Series([0] * len(df))

    def predict_proba(self, df: pd.DataFrame):
        """Return a dummy Series for testing purposes."""
        return pd.Series([0.1] * len(df))


class TestBaseADModel:
    """Tests for BaseADModel's fit() behavior and decision_function()."""

    def test_fit_drops_target_column(self, sample_df):
        """fit() should drop the target column before calling _fit()."""
        model = ConcreteADModel()
        model.fit(sample_df, target_col='anomaly')
        assert model.fitted_df is not None
        assert 'anomaly' not in model.fitted_df.columns
        assert 'feature1' in model.fitted_df.columns
        assert 'feature2' in model.fitted_df.columns

    def test_fit_ignores_missing_target_column(self, sample_df):
        """fit() should not raise if the target column is absent."""
        model = ConcreteADModel()
        df_no_label = sample_df.drop(columns=['anomaly'])
        model.fit(df_no_label, target_col='anomaly')
        assert model.fitted_df is not None

    def test_decision_function_delegates_to_predict_proba(self, features_df):
        """decision_function() should return the same result as predict_proba()."""
        model = ConcreteADModel()
        result = model.decision_function(features_df)
        expected = model.predict_proba(features_df)
        pd.testing.assert_series_equal(result, expected)

    def test_predict_raises_not_implemented(self):
        """Calling predict() on the raw base class should raise NotImplementedError."""
        model = BaseADModel()
        with pytest.raises(NotImplementedError):
            model.predict(pd.DataFrame())

    def test_predict_proba_raises_not_implemented(self):
        """Calling predict_proba() on the raw base class should raise NotImplementedError."""
        model = BaseADModel()
        with pytest.raises(NotImplementedError):
            model.predict_proba(pd.DataFrame())

    def test_fit_raises_not_implemented_via_base(self):
        """Calling fit() on the raw base class should raise via _fit()."""
        model = BaseADModel()
        with pytest.raises(NotImplementedError):
            model.fit(pd.DataFrame({'a': [1]}))


# ---------------------------------------------------------------------------
# PyCaretADModel – construction / validation
# ---------------------------------------------------------------------------
class TestPyCaretADModelInit:
    """Tests for PyCaretADModel's __init__() and parameter validation."""

    def test_valid_model_name(self):
        """Constructing with a valid model name should set attributes correctly."""
        model = PyCaretADModel(model_name='iforest')
        assert model.model_name == 'iforest'
        assert model.contamination == 0.05
        assert model.model is None

    def test_custom_contamination(self):
        """Constructing with a custom contamination should set it correctly."""
        model = PyCaretADModel(model_name='knn', contamination=0.1)
        assert model.contamination == 0.1

    def test_invalid_model_name_raises(self):
        """Constructing with an invalid model name should raise a ValueError."""
        with pytest.raises(ValueError, match='not allowed'):
            PyCaretADModel(model_name='invalid_model')

    def test_all_valid_model_names_accepted(self):
        """All model names in parameter_options should be accepted without error."""
        for name in PyCaretADModel.parameter_options['model_name']:
            model = PyCaretADModel(model_name=name)
            assert model.model_name == name

    def test_extra_kwargs_ignored(self):
        """Passing extra kwargs to __init__ should not raise an error."""
        model = PyCaretADModel(model_name='pca', unknown_param='x')
        assert model.model_name == 'pca'


# ---------------------------------------------------------------------------
# PyCaretADModel – predict / predict_proba before fitting
# ---------------------------------------------------------------------------
class TestPyCaretADModelNotFitted:
    """Tests for PyCaretADModel's predict() and predict_proba() behavior when not fitted."""

    def test_predict_raises_when_not_fitted(self, features_df):
        """predict() should raise a ValueError if the model has not been fitted."""
        model = PyCaretADModel(model_name='iforest')
        with pytest.raises(ValueError, match='not been fitted'):
            model.predict(features_df)

    def test_predict_proba_raises_when_not_fitted(self, features_df):
        """predict_proba() should raise a ValueError if the model has not been fitted."""
        model = PyCaretADModel(model_name='iforest')
        with pytest.raises(ValueError, match='not been fitted'):
            model.predict_proba(features_df)


# ---------------------------------------------------------------------------
# PyCaretADModel – fit / predict / predict_proba with mocked PyCaret
# ---------------------------------------------------------------------------
class TestPyCaretADModelWithMocks:
    """Test PyCaret-dependent methods by patching the pycaret imports."""

    @pytest.fixture(autouse=True)
    def patch_pycaret(self):
        """Patch pycaret's setup, create_model, and predict_model functions."""
        mock_model = MagicMock()
        with (
            patch('ml.ad.anomaly_detection.setup') as mock_setup,
            patch('ml.ad.anomaly_detection.create_model', return_value=mock_model) as mock_create,
            patch('ml.ad.anomaly_detection.predict_model') as mock_predict,
        ):
            self.mock_setup = mock_setup
            self.mock_create = mock_create
            self.mock_predict = mock_predict
            self.mock_model = mock_model
            yield

    def test_fit_calls_setup_and_create_model(self, sample_df):
        """fit() should call pycaret's setup() and create_model() with the correct parameters."""
        # Default contamination should be used if not overridden in kwargs
        model = PyCaretADModel(model_name='iforest')
        # _fit() is called by fit(), which should call setup() and create_model()
        model.fit(sample_df)
        # Verify setup() was called with the DataFrame and expected parameters
        self.mock_setup.assert_called_once()
        self.mock_create.assert_called_once_with('iforest', fraction=0.05)
        assert model.model is self.mock_model

    def test_fit_drops_target_column_before_pycaret(self, sample_df):
        """The DataFrame passed to pycaret's setup() should not contain the target column."""
        model = PyCaretADModel(model_name='lof')
        model.fit(sample_df, target_col='anomaly')

        # The DataFrame passed to setup() must not contain 'anomaly'
        call_kwargs = self.mock_setup.call_args
        passed_df = call_kwargs[1]['data']  # setup() is called with data=df
        assert 'anomaly' not in passed_df.columns

    def test_fit_updates_contamination_from_kwargs(self, sample_df):
        """If contamination is provided in kwargs to fit(), it should override the default."""
        model = PyCaretADModel(model_name='iforest')
        model.fit(sample_df, contamination=0.2)

        assert model.contamination == 0.2
        self.mock_create.assert_called_once_with('iforest', fraction=0.2)

    def test_predict_returns_anomaly_column(self, features_df):
        """predict() should return a Series with the anomaly labels."""
        # Set up the mock to return a DataFrame with 'Anomaly' and 'Anomaly_Score' columns
        expected = pd.Series([0, 1, 0], name='Anomaly')
        self.mock_predict.return_value = pd.DataFrame(
            {'Anomaly': expected, 'Anomaly_Score': [0.1, 0.9, 0.2]}
        )

        # Set the model to a mock so that predict() doesn't raise about not being fitted
        model = PyCaretADModel(model_name='iforest')
        model.model = self.mock_model

        # Call predict() and verify it returns the 'Anomaly' column as a Series
        result = model.predict(features_df)
        pd.testing.assert_series_equal(result, expected)

    def test_predict_proba_returns_anomaly_score_column(self, features_df):
        """predict_proba() should return a Series with the anomaly scores."""
        # Set up the mock to return a DataFrame with 'Anomaly' and 'Anomaly_Score' columns
        expected_scores = pd.Series([0.1, 0.9, 0.2], name='Anomaly_Score')
        self.mock_predict.return_value = pd.DataFrame(
            {'Anomaly': [0, 1, 0], 'Anomaly_Score': expected_scores}
        )

        # Set the model to a mock so that predict_proba() doesn't raise about not being fitted
        model = PyCaretADModel(model_name='iforest')
        model.model = self.mock_model

        # Call predict_proba() and verify it returns the 'Anomaly_Score' column as a Series
        result = model.predict_proba(features_df)
        pd.testing.assert_series_equal(result, expected_scores)

    def test_decision_function_equals_predict_proba(self, features_df):
        """decision_function() should return the same result as predict_proba()."""
        # Set up the mock to return a DataFrame with 'Anomaly' and 'Anomaly_Score' columns
        scores = pd.Series([0.3, 0.7, 0.1], name='Anomaly_Score')
        self.mock_predict.return_value = pd.DataFrame(
            {'Anomaly': [0, 1, 0], 'Anomaly_Score': scores}
        )

        # Set the model to a mock so that decision_function() doesn't raise about not being fitted
        model = PyCaretADModel(model_name='iforest')
        model.model = self.mock_model

        # Verify that decision_function() returns the same Series as predict_proba()
        pd.testing.assert_series_equal(
            model.decision_function(features_df), model.predict_proba(features_df)
        )
