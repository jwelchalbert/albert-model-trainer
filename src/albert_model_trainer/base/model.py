import logging
import warnings
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.base import BaseEstimator
from sklearn.base import clone as clone_model
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
import copy

from albert_model_trainer.base.callback import Callback, CallbackInvoker
from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.metrics import (
    Metric,
    NamedAggregatePerformanceMetrics,
    PerformanceMetrics,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
logger = logging.getLogger("albert.log")


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type, presumably a script
    except NameError:
        return False  # Probably a script


class ModelTrainer(CallbackInvoker):
    def __init__(
        self,
        hyperparameters: HyperParameterTuneSet | None = None,
        num_cv_folds: int = 5,
        evaluation_metric: str | Metric = "r2",
        random_state: int = 42,
        metrics: PerformanceMetrics | None = None,
        callbacks: List[Callback] | None = None,
        scaling_columns: List[int] | None = None,
        num_hyperopt_samples: int = 100,
    ):
        super().__init__(callbacks)
        self.hyperparameters = hyperparameters
        self.model: BaseEstimator | None = None
        self.num_cv_folds = num_cv_folds
        self.random_state = random_state
        self.optimal_parameters = {}
        self.scaling_columns = scaling_columns
        self.num_hyperopt_samples = num_hyperopt_samples

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

    def clone(self) -> "ModelTrainer":
        new_model_trainer = self.__class__(
            hyperparameters=self.hyperparameters,
            num_cv_folds=self.num_cv_folds,
            evaluation_metric=self.evaluation_metric,
            random_state=self.random_state,
            metrics=self.metrics,
            callbacks=self.callbacks,
            num_hyperopt_samples=self.num_hyperopt_samples,
        )
        new_model_trainer.model = clone_model(self.model)
        return new_model_trainer

    def set_custom_config(self, config: Dict[str, Any]) -> None:
        """
        Set the parameters of the model which are in the config dictionary.
            Call this prior to fit to use custom parameters instead of the default ones for the model,
            Override this function only if you need to implement custom logic, such as if your model,
            wraps an estimator and you need the config to set parameters on the wrapped estimator.
        """
        for param, val in config.items():
            setattr(self.model, param, val)

    def fit(self, X: Any, y: Any | None = None, config: Dict[str, Any] | None = None):
        """
        Fit the actual model -- optional config dictionary to set parameters on the model.
        """
        raise NotImplementedError

    def evaluate(self) -> PerformanceMetrics:
        raise NotImplementedError

    def set_hyperparameters(self, hyperparameters: HyperParameterTuneSet):
        self.hyperparameters = hyperparameters

    def name(self):
        return self.__class__.__name__

    def cross_validate(
        self,
        X: Any,
        y: Any | None = None,
        n_splits=5,
        random_state=42,
        display_fold_progress=False,
        generate_pred_obs_plots=False,
        encode_plots_base64=False,
    ) -> (
        NamedAggregatePerformanceMetrics
        # trunk-ignore(ruff/F821)
        | Tuple[
            NamedAggregatePerformanceMetrics, Union[bytes, "matplotlib.figure.Figure"]
        ]
    ):
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

        all_metrics = NamedAggregatePerformanceMetrics()
        all_metrics.add_metric_group("train")
        all_metrics.add_metric_group("val")

        plotdata: list[Tuple[np.ndarray, np.ndarray]] = []

        folds = kf.split(X)
        if display_fold_progress:
            folds = tqdm(folds, desc="Training on Folds")

        for train_index, val_index in folds:
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Create the scalar if we were told to scale certain columns
            if self.scaling_columns is not None:
                # logger.info(f"Scaling Columns {self.scaling_columns}")
                scaler = ColumnTransformer(
                    [("scalar", StandardScaler(), self.scaling_columns)],
                    remainder="passthrough",
                )

                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            self.fit(X_train, y_train)

            y_train_pred = self.model.predict(X_train)
            y_val_pred = self.model.predict(X_val)

            train_metrics_set = self.metrics.clone()
            train_metrics_set.evaluate_all(y_train, y_train_pred)
            all_metrics.add_metrics("train", train_metrics_set)

            val_metrics_set = self.metrics.clone()
            val_metrics_set.evaluate_all(y_val, y_val_pred)
            all_metrics.add_metrics("val", val_metrics_set)

            if generate_pred_obs_plots:
                # Store the data for generating the plot later
                plotdata.append((y_val, y_val_pred))

        if generate_pred_obs_plots:
            import matplotlib
            import matplotlib.figure

            matplotlib.use("Agg")
            import base64
            import io

            import matplotlib.pyplot as plt

            # Generate a prediction vs observation plot for all folds
            fig = plt.figure(figsize=(10, 10))
            for i, (x, y) in enumerate(plotdata):
                plt.scatter(x, y, alpha=0.2, label=f"Fold {i}")

            # Get the current axes, so we can add a line to it
            ax = plt.gca()

            # Get the limits of the current axes
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Find the range of both axes
            range_all = [np.min([xlim, ylim]), np.max([xlim, ylim])]

            # Plot the diagonal line
            plt.plot(range_all, range_all, "k--")

            plt.ylabel("Predicted Values")
            plt.xlabel("Observed Values")
            plt.title("Prediction vs Observation Plot")
            plt.legend()

            if encode_plots_base64:
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)

                image_64 = base64.b64encode(buf.read())
                figdata = image_64

            else:
                figdata = fig

            return all_metrics, figdata

        return all_metrics

    def fit_tune(self, X: Any, y: Any | None = None):
        """
        Carry out the actual ray tune operation on the model.
        """
        if self.model is None:
            raise ValueError("no model instance is set -- nothing to tune")

        # Call Setup on the callbacks
        self.trigger_callback("setup", args={"trainer": self})

        def objective(config) -> None:
            # Get a clone of the trainer so we can configure the underlying model
            # without affecting other tune instances
            trainer = self.clone()
            trainer.set_custom_config(config)
            print(f"Metric: {trainer.evaluation_metric}")

            trainer.trigger_callback("on_tune_step_begin", args={"trainer": trainer})

            metrics: NamedAggregatePerformanceMetrics = trainer.cross_validate(
                X, y, self.num_cv_folds, self.random_state, False, False, False
            )

            self.trigger_callback(
                "on_tune_train_end",
                {"trainer": trainer, "metrics": metrics.get_metric_group("train")},
            )
            self.trigger_callback(
                "on_tune_validation_end",
                {"trainer": trainer, "metrics": metrics.get_metric_group("val")},
            )

            avg_eval_metric = metrics.get_group_metric_avg(
                "val", self.evaluation_metric
            )
            avg_eval_std = metrics.get_group_metric_std("val", self.evaluation_metric)

            tune.report(
                **{
                    self.evaluation_metric: avg_eval_metric,
                    f"{self.evaluation_metric}_std": avg_eval_std,
                }
            )

            self.trigger_callback("on_tune_step_complete", {"trainer": trainer})

        if is_notebook():
            logger.info("Starting Notebook Reporter")
            reporter = tune.JupyterNotebookReporter(
                metric_columns=["loss", "training_iteration", self.evaluation_metric]
            )
        else:
            logger.info("Starting CLI Reporter")
            reporter = tune.CLIReporter(
                metric_columns=["loss", "training_iteration", self.evaluation_metric]
            )

        hyperopt_search = HyperOptSearch(
            metric=self.evaluation_metric,
            mode=self.metrics.get_metric_obj(self.evaluation_metric).optimal_mode(),
        )
        hyperopt_search = ConcurrencyLimiter(hyperopt_search, max_concurrent=4)

        self.trigger_callback("on_ray_pre_init", {"trainer": self})

        ray.init(_memory=5000000000, object_store_memory=2000000000, num_cpus=2)
        analysis = tune.run(
            objective,
            resources_per_trial={"cpu": 1},
            config=self.hyperparameters.parameters,
            search_alg=hyperopt_search,
            progress_reporter=reporter,
            num_samples=100,
            raise_on_failed_trial=False,
        )

        self.trigger_callback("on_ray_pre_shutdown", {"trainer": self})
        ray.shutdown()

        tdata = analysis.get_best_trial(
            metric=self.evaluation_metric,
            mode=self.metrics.get_metric_obj(self.evaluation_metric).optimal_mode(),
        )
        best_params = analysis.get_best_config(
            metric=self.evaluation_metric,
            mode=self.metrics.get_metric_obj(self.evaluation_metric).optimal_mode(),
        )

        self.best_params = best_params
        return tdata.last_result, best_params
