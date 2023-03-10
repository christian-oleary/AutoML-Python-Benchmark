"""
Abstract Classes
"""

from abc import ABC, abstractmethod

class Forecaster(ABC):
    """Abstract Forecaster"""

    @abstractmethod
    def forecast(self):
        """Run forecasting code. API to be defined."""
        pass
