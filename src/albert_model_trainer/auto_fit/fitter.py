from typing import Any, Dict, Iterable, Tuple, List

from sklearn.base import BaseEstimator, RegressorMixin

from albert_model_trainer.base.callback import Callback, CallbackInvoker
from albert_model_trainer.base.metrics import Metric, PerformanceMetrics
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import ModelConfigurationBase
from albert_model_trainer.models.model_iterator import get_all_model_trainers
from sklearn.base import clone as clone_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


class MultiModelRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, models: Iterable[RegressorMixin]) -> None:
        self.models: list[RegressorMixin] = []
        for m in models:
            self.models.append(clone_model(m))

    def fit(self, X: Any, y: Any | None = None) -> "MultiModelRegressor":
        for i, m in enumerate(self.models):
            m.fit(X, y[:, i])
        return self

    def predict(self, X: Any) -> Any:
        results = []
        for m in self.models:
            results.append(m.predict(X))
        return np.column_stack(results)


class ModelRegistry:
    def __init__(self, num_outputs, sort_max_first) -> None:
        self.num_outputs = num_outputs
        self.params: Dict[
            int,
            List[
                Tuple[
                    str, float, Dict[str, float], Dict[str, Any], ModelConfigurationBase
                ]
            ],
        ] = {i: [] for i in range(num_outputs)}
        self.sort_max_first = sort_max_first

    def add_result(
        self,
        output_idx: int,
        name: str,
        score: float,
        metrics: dict,
        parameters: dict[str, Any],
        model_config: ModelConfigurationBase,
    ) -> None:
        if output_idx > self.num_outputs:
            raise ValueError(
                f"requested output idx is greater than the expected number of outputs -- got [{output_idx}] -- exp [{self.num_outputs-1}] max"
            )
        self.params[output_idx].append((name, score, metrics, parameters, model_config))

    def get_best_params(self, return_max: bool) -> list[Any]:
        best_all = []
        for i in range(self.num_outputs):
            best_params = self.params[i]
            best_params = sorted(best_params, key=lambda x: x[1], reverse=return_max)
            best_all.append(best_params[0])
        return best_all

    def get_sorted_model_info(self, max: bool) -> list[Any]:
        bsets = []
        for i in range(self.num_outputs):
            best_params = self.params[i]
            best_params = sorted(best_params, key=lambda x: x[1], reverse=max)
            bsets.append(best_params)
        # Create tuples of model groups that are sorted by the score
        bsets = list(zip(*bsets))
        return bsets

    def get_model_trainer(self, model_name: str) -> ModelTrainer:
        all_models = get_all_model_trainers()

        # See if we can find the requested model
        model_ref: ModelTrainer | None = None
        for mm in all_models:
            if (
                mm.name().lower() == model_name.lower()
                or mm.modelName().lower() == model_name.lower()
            ):
                # We found the requested model -- lets grab a reference to it
                model_ref = mm.clone()

        if model_ref is None:
            raise TypeError(f"unknown model requested -- [{model_name}]")

        return model_ref

    def get_model_pipeline(
        self,
        model_name: str | Iterable[str],
        model_params: dict[str, Any] | Iterable[dict[str, Any]],
        scaling_columns: Iterable[int] | List[Iterable[int] | None] | None = None,
    ):
        if isinstance(model_name, str) and isinstance(model_params, dict):
            model_trainer = self.get_model_trainer(model_name)
            # Setup the model with the given parameters -- this will configure the underlying model
            model_trainer.set_custom_config(model_params)
            final_model = clone_model(model_trainer.model)
        elif isinstance(model_name, Iterable) or isinstance(model_params, Iterable):
            if not isinstance(model_params, Iterable) or not isinstance(
                model_name, Iterable
            ):
                raise TypeError(
                    "model_name and model_params should both be lists if one or the other is"
                )

            if len(model_name) != len(model_params):
                raise ValueError(
                    "model_name and model_params should both be the same length"
                )

            # We have to build a pipeline that has multiple models as output
            model_refs = []
            for i, mname in enumerate(model_name):
                model_trainer = self.get_model_trainer(mname)
                model_trainer.set_custom_config(model_params[i])

                # Now the model has the correct hyperparameters set,
                # store a reference to it for our multimodel regressor
                model_refs.append(model_trainer.model)

            # Now create a multi model regressor out of these models
            final_model = MultiModelRegressor(model_refs)
        else:
            raise TypeError("invalid model_name, model_params type combination")

        pipeline: Pipeline
        if scaling_columns is not None:
            pipeline = Pipeline(
                [
                    ("scalar", StandardScaler()),
                    ("model", final_model),
                ]
            )
        else:
            pipeline = Pipeline([("model", final_model)])

        return pipeline

    def get_nth_top_model_pipeline(
        self, nth_model=0, return_meta_data=False
    ) -> tuple[Pipeline, Any, Any, Any] | Pipeline:
        if nth_model >= len(self.params[0]):
            raise ValueError(
                f"n was too large, we only have {len(self.params[0])} models so n must be less than that"
            )

        model_info = self.get_sorted_model_info(self.sort_max_first)
        param_set = model_info[nth_model]

        # Set Expected Types Here
        model_configs: List[ModelConfigurationBase]
        model_names: List[str]
        model_scores: List[float]
        model_params: List[dict[str, Any]]

        model_names, model_scores, model_metrics, model_params, model_configs = zip(
            *param_set
        )

        if len(param_set) > 1:
            pipeline = self.get_model_pipeline(
                model_names, model_params, [x.scaling_columns for x in model_configs]
            )
        else:
            pipeline = self.get_model_pipeline(
                model_names[0], model_params[0], model_configs[0].scaling_columns
            )

        if return_meta_data:
            return pipeline, model_names, model_scores, model_metrics
        else:
            return pipeline


class SklearnAutoRegressor(BaseEstimator, RegressorMixin, CallbackInvoker):
    result_tracker: ModelRegistry

    def __init__(
        self,
        evaluation_metric: str | Metric = "r2",
        metrics: str | list[str] | list[Metric] | PerformanceMetrics | None = None,
        num_cv_folds: int = 5,  # Number of cross validated folds to perform when calculating the performance metric
        models: Dict[str, ModelTrainer] | list[ModelTrainer] | None = None,
        callbacks: list[Callback] | None = None,
        random_state=42,
        num_hyperopt_samples: int = 100,
        scaling_columns: list[int] | None = None,
    ) -> None:
        BaseEstimator.__init__(self)
        RegressorMixin.__init__(self)
        CallbackInvoker.__init__(self, callbacks)

        self.random_state = random_state
        self.num_hyperopt_samples = num_hyperopt_samples
        self.scaling_columns = scaling_columns

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
            self.metrics = PerformanceMetrics()
            # If the user passed in a list of strings or metric objects then
            # we add them directly -- the PerformanceMetrics object will handle type checks
            if len(metrics) > 0:
                for metric in metrics:
                    self.metrics.add_metric(metric)
        elif isinstance(metrics, PerformanceMetrics):
            # If they pass in a prepopulated PM object, then we just use that
            self.metrics = metrics
        else:
            self.metrics = PerformanceMetrics()
            # Empty Set of Metrics -- we will add the evaluation metric next so there is atleast one

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

    def add_model(self, model: ModelTrainer, raise_on_exists=False) -> None:
        if model.name() not in self.models:
            # We override the evaluation metric on the model with the one requested at the high level
            # this makes sure everyone is operating on the same space
            model.config.num_cv_folds = self.num_cv_folds
            model.config.evaluation_metric = self.evaluation_metric
            model.config.random_state = self.random_state
            model.config.metrics = self.metrics
            model.config.num_hyperopt_samples = self.num_hyperopt_samples
            model.config.scaling_columns = self.scaling_columns
            model.config.callbacks = self.callbacks

            # We also explicitly the callbacks on the model itself which is the invoker,
            # normally this is done in the constructor but we aren't explicilty calling the
            # constructor here
            if model.callbacks is not None:
                model.callbacks.extend(self.callbacks)
            else:
                model.callbacks = self.callbacks

            self.models[model.name()] = model
        else:
            if raise_on_exists:
                raise NameError(
                    f"Model type {model.name()} is already registered to be trained"
                )

    def fit(self, X: Any, y: Any | None = None) -> list[Any]:
        if y is None:
            raise AssertionError("cannot regress without a target")
        if not (len(y.shape) <= 2):
            raise AssertionError(
                "only tensors of shape [B, E] are allowed as outputs where E >= 1"
            )

        self.multi_output = False
        if (len(y.shape) > 1) and (y.shape[-1] > 1):
            self.multi_output = True

        self.result_tracker = ModelRegistry(
            y.shape[-1] if self.multi_output else 1,
            self.metrics.get_metric_obj(self.evaluation_metric).optimal_mode() == "max",
        )

        # We will iterate through each model and do a full hyperparameter tune on it
        for idx, (_, model_trainer) in enumerate(self.models.items()):
            self.trigger_callback(
                "on_tune_start",
                {
                    "trainer": model_trainer,
                    "model_idx": idx,
                    "num_features": y.shape[-1] if self.multi_output else 1,
                },
            )

            # If we have multiple outputs then we need to build one model for each output
            if self.multi_output:
                self.trigger_callback(
                    "on_tune_multi_output_start",
                    {"trainer": model_trainer, "total_outputs": y.shape[-1]},
                )
                for i in range(y.shape[-1]):
                    tdata, best_params = model_trainer.fit_tune(X, y[:, i])
                    self.result_tracker.add_result(
                        i,
                        model_trainer.modelName(),
                        tdata[self.evaluation_metric],
                        tdata,
                        best_params,
                        model_trainer.config,
                    )

                    # In multi output cases indicate when we finish one of the outputs
                    self.trigger_callback(
                        "on_tune_multi_output_end",
                        {
                            "trainer": model_trainer,
                            "output_num": i,
                            "total_outputs": y.shape[-1],
                        },
                    )
            else:
                tdata, best_params = model_trainer.fit_tune(X, y)
                self.result_tracker.add_result(
                    0,
                    model_trainer.modelName(),
                    tdata[self.evaluation_metric],
                    tdata,
                    best_params,
                    model_trainer.config,
                )

            self.trigger_callback(
                "on_tune_end", {"trainer": model_trainer, "model_idx": idx}
            )

        return self.result_tracker.get_best_params(
            self.metrics.get_metric_obj(self.evaluation_metric).optimal_mode() == "max"
        )
