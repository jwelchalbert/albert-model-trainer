from typing import List
from .hyperparameter import HyperParameterTuneSet
from .metrics import Metric, PerformanceMetrics
from .callback import Callback


def validate_config_type(config, expected_hyperparameter_type):
    if not isinstance(config.hyperparameters, expected_hyperparameter_type):
        raise TypeError(
            f"hyperparameter type needs to match the model type -- got [{type(config.hyperparameters)}] -- exp [{expected_hyperparameter_type}]"
        )


class ModelConfigurationBase:
    def __init__(
        self,
        hyperparameters: HyperParameterTuneSet,
        num_cv_folds: int = 5,
        evaluation_metric: str | Metric = "r2",
        random_state: int = 42,
        metrics: PerformanceMetrics | None = None,
        callbacks: List[Callback] | None = None,
        num_hyperopt_samples: int = 100,
        scaling_columns: List[int] | None = None,
    ) -> None:
        self.hyperparameters = hyperparameters
        self.num_cv_folds = num_cv_folds
        self.random_state = random_state
        self.scaling_columns = scaling_columns
        self.num_hyperopt_samples = num_hyperopt_samples
        self.evaluation_metric = evaluation_metric
        self.callbacks = callbacks

        if metrics is None:
            self.metrics = PerformanceMetrics()
        else:
            self.metrics = metrics

        if isinstance(evaluation_metric, str):
            if not self.metrics.has_metric(evaluation_metric):
                self.metrics.add_metric(evaluation_metric)
            self.evaluation_metric = evaluation_metric

        elif isinstance(evaluation_metric, Metric):
            if not self.metrics.has_metric(evaluation_metric.shortnames()[0]):
                self.metrics.add_metric(evaluation_metric.shortnames()[0])

            # Store the name of the metric so we know which one to call back later
            self.evaluation_metric = evaluation_metric.shortnames()[0]
