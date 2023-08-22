from typing import Any, Dict

from sklearn.base import BaseEstimator, RegressorMixin

from albert_model_trainer.base.callback import Callback, CallbackInvoker
from albert_model_trainer.base.metrics import Metric, PerformanceMetrics
from albert_model_trainer.base.model import ModelTrainer


class ResultTracker:
    def __init__(self, num_outputs) -> None:
        self.num_outputs = num_outputs
        self.params = {i: [] for i in range(num_outputs)}

    def add_result(
        self, output_idx: int, name: str, score: float, parameters: dict[str, Any]
    ):
        if output_idx > self.num_outputs:
            raise ValueError(
                f"requested output idx is greater than the expected number of outputs -- got [{output_idx}] -- exp [{self.num_outputs-1}] max"
            )
        self.params[output_idx].append((name, score, parameters))

    def get_best_params(self, return_max: bool):
        best_all = []
        for i in range(self.num_outputs):
            best_params = self.params[i]
            best_params = sorted(best_params, key=lambda x: x[1], reverse=return_max)
            best_all.append(best_params[0])
        return best_all


class SklearnAutoRegressor(BaseEstimator, RegressorMixin, CallbackInvoker):
    def __init__(
        self,
        evaluation_metric: str | Metric = "r2",
        metrics: str | list[str] | list[Metric] | PerformanceMetrics | None = None,
        num_cv_folds: int = 5,  # Number of cross validated folds to perform when calculating the performance metric
        models: Dict[str, ModelTrainer] | list[ModelTrainer] | None = None,
        callbacks: list[Callback] | None = None,
    ) -> None:
        BaseEstimator.__init__(self)
        RegressorMixin.__init__(self)
        CallbackInvoker.__init__(self, callbacks)

        self.num_cv_folds = num_cv_folds

        if isinstance(metrics, str):
            self.metrics = PerformanceMetrics()
            # A single metric or list of metrics which are comma delimited
            # Obviously this doesn't work for metrics which have required extra parameters
            # this only works for metrics which have default extra parameters or no extra parameters
            fields = metrics.split(",")
            for metric_name in fields:
                self.metrics.add_metric(metric_name)
        elif isinstance(metrics, list):
            # If the user passed in a list of strings or metric objects then
            # we add them directly -- the PerformanceMetrics object will handle type checks
            if len(metrics) > 0:
                for metric in metrics:
                    self.metrics.add_metric(metric)
        elif isinstance(metrics, PerformanceMetrics):
            # If they pass in a prepopulated PM object, then we just use that
            self.metrics = metrics
        else:
            self.metrics = (
                PerformanceMetrics()
            )  # Empty Set of Metrics -- we will add the evaluation metric next so there is atleast one

        if isinstance(evaluation_metric, str):
            if not self.metrics.has_metric(evaluation_metric):
                self.metrics.add_metric(evaluation_metric)
            self.evaluation_metric = evaluation_metric

        elif isinstance(evaluation_metric, Metric):
            metric_name = evaluation_metric.shortnames()[0]
            if not self.metrics.has_metric(metric_name):
                self.metrics.add_metric(
                    evaluation_metric
                )  # Add this to the list if it wasn't already there
            self.evaluation_metric = metric_name

        # If they passed in a set of models then we add them here
        if models is not None:
            if isinstance(models, Dict):
                self.models = models
            elif isinstance(models, list):
                self.models = {}
                for mm in models:
                    self.add_model(mm, True)
        else:
            self.models = {}

    def add_model(self, model: ModelTrainer, raise_on_exists=False):
        if model.name() not in self.models:
            # We override the evaluation metric on the model with the one requested at the high level
            # this makes sure everyone is operating on the same space
            model.config.metrics = self.metrics
            model.config.evaluation_metric = self.evaluation_metric
            model.callbacks = self.callbacks
            self.models[model.name()] = model
        else:
            if raise_on_exists:
                raise NameError(
                    f"Model type {model.name()} is already registered to be trained"
                )

    def fit(self, X: Any, y: Any | None = None):
        if y is None:
            raise AssertionError("cannot regress without a target")
        if not (len(y.shape) <= 2):
            raise AssertionError(
                "only tensors of shape [B, E] are allowed as outputs where E >= 1"
            )

        self.multi_output = False
        if (len(y.shape) > 1) and (y.shape[-1] > 1):
            self.multi_output = True

        self.best_params = ResultTracker(y.shape[-1])

        # We will iterate through each model and do a full hyperparameter tune on it
        for idx, (model_name, model_trainer) in enumerate(self.models.items()):
            self.trigger_callback(
                "on_tune_start",
                {
                    "trainer": model_trainer,
                    "model_idx": idx,
                    "num_features": y.shape[-1] if self.multi_output else 1,
                },
            )

            # If we have multiple outputs then we need to build one model for each output
            for i in range(y.shape[-1]):
                tdata, best_params = model_trainer.fit_tune(X, y[:, i])
                self.best_params.add_result(
                    i, model_name, tdata[self.evaluation_metric], best_params
                )
                if self.multi_output:
                    # In multi output cases indicate when we finish one of the outputs
                    self.trigger_callback(
                        "on_tune_multi_output_end",
                        {"trainer": model_trainer, "output_num": i},
                    )

            self.trigger_callback(
                "on_tune_end", {"trainer": model_trainer, "model_idx": idx}
            )

            return self.best_params.get_best_params(
                self.metrics.get_metric_obj(self.evaluation_metric).optimal_mode()
                == "max"
            )
