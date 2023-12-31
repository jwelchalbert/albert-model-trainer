import inspect
from typing import Any, Dict, List

import numpy as np

import albert_model_trainer.base.metrics.metrics as metrics

from .metrics import Metric

import logging

logger = logging.getLogger("albert.log")


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
    def __init__(self, metrics_to_eval: List[Metric] | List[str] | None = None):
        if metrics_to_eval is None:
            metrics_to_eval = []
        self.metrics: List[Metric] = []
        for m in metrics_to_eval:
            self.add_metric(m)
        self.results = {}

    def has_metric(self, metric_name: str) -> bool:
        for m in self.metrics:
            if metric_name.lower() in m.shortnames():
                return True

        return False

    def get_metric(self, metric_name: str) -> float:
        logger.debug(f"Searching for metric {metric_name}")
        for m in self.metrics:
            logger.debug(f"{m} -- {m.shortnames()}")
            if metric_name.lower() in m.shortnames():
                logger.debug(f"Found Metric for {metric_name} as {str(m)}")
                if str(m) in self.results:
                    return self.results[str(m)]
                else:
                    raise KeyError(
                        f"no metric named {metric_name} has been calculated yet -- be sure you call `evaluate_all` prior to trying to get the metric"
                    )
        else:
            raise KeyError(
                f"no metric named {metric_name} has been registered as a performance metric"
            )

    def get_metric_obj(self, metric_name: str) -> Metric:
        for m in self.metrics:
            if metric_name.lower() in m.shortnames():
                return m

        return None

    def clone(self) -> "PerformanceMetrics":
        new_metrics = PerformanceMetrics(self.metrics)

        for metric_name, result in self.results.items():
            new_metrics.results[metric_name] = result

        return new_metrics

    def add_metric(self, metric: str | Metric | list[Metric] | list[str], **kwargs):
        logger.debug(
            f"adding metric {str(metric) if isinstance(metric, Metric) else metric}"
        )

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
            val = metric.evaluate(true_values, predictions)
            logger.debug(f"Computing {str(metric)} -- {val}")
            self.results[str(metric)] = val

    def print_summary(self):
        for metric_name, result in self.results.items():
            logger.info(f"{metric_name}: {result}")


class AggregatePerformanceMetrics:
    def __init__(self) -> None:
        self.metrics: list[PerformanceMetrics] = []

    def add_metrics(self, metrics: PerformanceMetrics):
        self.metrics.append(metrics.clone())

    def get_metric_vals(self, metric_name: str):
        vals = []
        for pm in self.metrics:
            try:
                val = pm.get_metric(metric_name)
                if (val is None) or (np.isnan(val)):
                    print(
                        "One of the results was nan, skipping but this may artifically skew the results"
                    )
                    continue
                vals.append(val)
            except KeyError as e:
                logger.error(f"Unknown key requested {(str(e))}")
                continue
        return vals

    def get_metric_avg(self, metric_name: str) -> float:
        return np.mean(self.get_metric_vals(metric_name))

    def get_metric_std(self, metric_name: str) -> float:
        return np.std(self.get_metric_vals(metric_name))

    def get_available_metrics(self):
        metrics = []
        if self.metrics is not None:
            for pm in self.metrics:
                for m in pm.metrics:
                    metrics.append(m.shortnames()[0])
        return metrics


class NamedAggregatePerformanceMetrics:
    def __init__(self) -> None:
        self.metrics: Dict[str, AggregatePerformanceMetrics] = {}

    def add_metric_group(
        self, group: str, metrics: AggregatePerformanceMetrics | None = None
    ):
        if metrics is None:
            self.metrics[group] = AggregatePerformanceMetrics()
        else:
            self.metrics[group] = metrics

    def add_metrics(self, group: str, metrics: PerformanceMetrics):
        self.metrics[group].add_metrics(metrics)

    def get_group_metric_avg(self, group: str, metric_name: str) -> float:
        return self.metrics[group].get_metric_avg(metric_name)

    def get_group_metric_std(self, group: str, metric_name: str) -> float:
        return self.metrics[group].get_metric_std(metric_name)

    def get_metric_group(self, group: str) -> AggregatePerformanceMetrics:
        return self.metrics.get(group)

    def get_available_metrics(self) -> list[str]:
        a_metrics = []
        if self.metrics is not None:
            for m in self.metrics:
                a_metrics.extend(self.metrics[m].get_available_metrics())

        return a_metrics
