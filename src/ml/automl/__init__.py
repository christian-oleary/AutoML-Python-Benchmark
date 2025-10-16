"""AutoML module."""

from abc import ABC, abstractmethod
import pandas as pd
from ml.configuration import Configuration


class AutoMLEngine(ABC):
    """Abstract base class for AutoML engines."""

    config: Configuration

    def __init__(self, config: Configuration):
        self.config = config

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series | None = None,
        **kwargs,
    ):
        """Implement training logic."""

    @abstractmethod
    def predict(self, X_test: pd.DataFrame) -> pd.Series:
        """Make predictions."""
