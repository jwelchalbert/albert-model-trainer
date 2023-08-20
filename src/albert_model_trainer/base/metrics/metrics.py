from typing import Any, Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


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


class PerformanceMetrics:
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
        self.results = {}

    def evaluate_all(self, true_values: Any, predictions: Any):
        for metric in self.metrics:
            self.results[str(metric)] = metric.evaluate(true_values, predictions)

    def print_summary(self):
        for metric_name, result in self.results.items():
            print(f"{metric_name}: {result}")


class AccuracyMetric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return accuracy_score(true_values, predictions)

    def shortnames(self) -> list[str]:
        return ["acc", "accuracy"]

    def __str__(self) -> str:
        return "Accuracy"


class PrecisionMetric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return precision_score(true_values, predictions)

    def shortnames(self) -> list[str]:
        return ["prec", "precision"]

    def __str__(self) -> str:
        return "Precision"


class MAEMetric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return mean_absolute_error(true_values, predictions)

    def shortnames(self) -> list[str]:
        return ["mean absolute error", "mae"]

    def __str__(self) -> str:
        return "Mean Absolute Error"


class MSEMetric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return mean_squared_error(true_values, predictions)

    def shortnames(self) -> list[str]:
        return ["mse", "mean squared error"]

    def __str__(self) -> str:
        return "Mean Squared Error"


class R2Metric(Metric):
    def evaluate(self, true_values: Any, predictions: Any) -> float:
        return r2_score(true_values, predictions)

    def shortnames(self) -> list[str]:
        return ["r2", "r^2", "r squared", "rsq"]

    def __str__(self) -> str:
        return "R^2 Score"
