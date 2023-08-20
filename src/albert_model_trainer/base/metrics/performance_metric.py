import inspect
import albert_model_trainer.base.metrics as metrics
from metrics import Metric
from typing import List, Tuple, Dict, Any


def generate_metric_mapping():
    """
    Dynamically generate a mapping from metric names/aliases (strings) to metric classes using the `shortnames` method.

    :return: Dictionary mapping metric names/aliases to their respective classes.
    """
    mapping = {}

    # Get all classes defined in the metrics_module
    for _, obj in inspect.getmembers(metrics):
        if inspect.isclass(obj) and issubclass(obj, Metric) and obj != Metric:
            # Use the shortnames method to fetch all acceptable names/aliases for the metric
            for name in obj().shortnames():
                mapping[name] = obj

    return mapping


METRIC_MAPPING = generate_metric_mapping()


class PerformanceMetrics:
    def __init__(self, metrics: List[Metric]):
        self.metrics = metrics
        self.results = {}

    def add_metric(self, metric: str | Metric | list[Metric] | list[str], **kwargs):
        def add_by_name(metric):
            if metric in METRIC_MAPPING:
                self.metrics.append(METRIC_MAPPING[metric](**kwargs))
            else:
                raise ValueError(f"unknown metric type requested {metric}")
            
        if isinstance(metric, str):
            add_by_name(metric)
        elif isinstance(metric, Metric):
            self.metrics.append(metric)
        elif isinstance(metric, list):
            for mm in metric:
                if isinstance(mm, str):
                    add_by_name(mm)
                elif isinstance(mm, Metric):
                    self.metrics.append(mm)
                else:
                    raise TypeError(
                        f"unknown type {type(mm)} -- cannot use as a metric"
                    )
        else:
            raise TypeError(f"unknown type {type(metric)} -- cannot use as a metric")

    def evaluate_all(self, true_values: Any, predictions: Any):
        for metric in self.metrics:
            self.results[str(metric)] = metric.evaluate(true_values, predictions)

    def print_summary(self):
        for metric_name, result in self.results.items():
            print(f"{metric_name}: {result}")
