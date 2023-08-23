from typing import Any

from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
)

from math import sqrt

METRIC_MODE_MIN = "min"
METRIC_MODE_MAX = "max"


class Metric:
    def evaluate(self, true_values: Any, predictions: Any, **kwargs) -> float:
        raise NotImplementedError(
            "The evaluate method should be overridden in the subclass."
        )

    def shortnames(self) -> list[str]:
        raise NotImplementedError(
            "The shortnames method should be overriden in the subclass."
        )

    def __str__(self) -> str:
        raise NotImplementedError(
            "The __str__ method should be overridden in the subclass."
        )

    def optimal_mode(self) -> str:
        raise NotImplementedError(
            "The optimal_mode method should be overridden in the subclass."
        )


class AccuracyMetric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return accuracy_score(true_values, predictions)

    def shortnames(self) -> list[str]:
        return ["acc", "accuracy"]

    def __str__(self) -> str:
        return "Accuracy"

    def optimal_mode(self) -> str:
        return METRIC_MODE_MAX


class PrecisionMetric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return precision_score(true_values, predictions)

    def shortnames(self) -> list[str]:
        return ["prec", "precision"]

    def __str__(self) -> str:
        return "Precision"

    def optimal_mode(self) -> str:
        return METRIC_MODE_MAX


class RootMeanSquaredErrorMetric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return sqrt(mean_squared_error(true_values, predictions))

    def shortnames(self) -> list[str]:
        return ["rmse", "root mean squared error"]

    def __str__(self) -> str:
        return "Root Mean Squared Error"

    def optimal_mode(self) -> str:
        return METRIC_MODE_MIN


class MAEMetric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return mean_absolute_error(true_values, predictions)

    def shortnames(self) -> list[str]:
        return ["mean absolute error", "mae"]

    def __str__(self) -> str:
        return "Mean Absolute Error"

    def optimal_mode(self) -> str:
        return METRIC_MODE_MIN


class MSEMetric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return mean_squared_error(true_values, predictions)

    def shortnames(self) -> list[str]:
        return ["mse", "mean squared error"]

    def __str__(self) -> str:
        return "Mean Squared Error"

    def optimal_mode(self) -> str:
        return METRIC_MODE_MIN


class R2ScoreMetric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return r2_score(true_values, predictions)

    def shortnames(self) -> list[str]:
        return ["r2", "r^2", "r squared", "rsq"]

    def __str__(self) -> str:
        return "R^2 Score"

    def optimal_mode(self) -> str:
        return METRIC_MODE_MAX
