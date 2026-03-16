"""This module provides a class to perform classification and regression using auto-sklearn."""

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


class AutoSklearnModel:
    """A class to perform classification and regression using auto-sklearn."""

    def __init__(self, time_left_for_this_task=300, per_run_time_limit=30, seed=42):
        """Initialize AutoSklearn models with time constraints."""
        self.time_left_for_this_task = time_left_for_this_task
        self.per_run_time_limit = per_run_time_limit
        self.seed = seed
        self.classifier = None
        self.regressor = None

    def train_classifier(self, X, y, test_size=0.2):
        """Train an AutoSklearn classifier."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed
        )

        self.classifier = AutoSklearnClassifier(
            time_left_for_this_task=self.time_left_for_this_task,
            per_run_time_limit=self.per_run_time_limit,
            seed=self.seed,
        )
        self.classifier.fit(X_train, y_train)
        predictions = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def train_regressor(self, X, y, test_size=0.2):
        """Train an AutoSklearn regressor."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed
        )

        self.regressor = AutoSklearnRegressor(
            time_left_for_this_task=self.time_left_for_this_task,
            per_run_time_limit=self.per_run_time_limit,
            seed=self.seed,
        )
        self.regressor.fit(X_train, y_train)
        predictions = self.regressor.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

    def predict_classifier(self, X):
        """Make predictions using the trained classifier."""
        if self.classifier is None:
            raise ValueError("Classifier not trained yet. Call train_classifier first.")
        return self.classifier.predict(X)

    def predict_regressor(self, X):
        """Make predictions using the trained regressor."""
        if self.regressor is None:
            raise ValueError("Regressor not trained yet. Call train_regressor first.")
        return self.regressor.predict(X)
