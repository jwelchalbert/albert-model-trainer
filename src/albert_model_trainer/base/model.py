import warnings
from typing import Any, Dict, Tuple, Union
import logging

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

from albert_model_trainer.base.callback import CallbackInvoker
from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.metrics import NamedAggregatePerformanceMetrics
from albert_model_trainer.base.model_config import ModelConfigurationBase

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
        config: ModelConfigurationBase,
    ):
        super().__init__(config.callbacks)
        self.config = config
        self.model: BaseEstimator | None = None
        self.optimal_parameters = {}

    def clone(self) -> "ModelTrainer":
        new_model_trainer = self.__class__(self.config)
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

    def fit(self, X: Any, y: Any | None = None):
        """
        Fit the actual model -- optional config dictionary to set parameters on the model.
        """
        if self.model is not None:
            self.model.fit(X, y)

    def set_hyperparameters(self, hyperparameters: HyperParameterTuneSet):
        self.hyperparameters = hyperparameters

    def name(self):
        return self.__class__.__name__

    def modelName(self):
        if self.model is not None:
            return self.model.__class__.__name__
        return "None"

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
        | Tuple[
            NamedAggregatePerformanceMetrics,
            # trunk-ignore(ruff/F821)
            Union[bytes, "matplotlib.figure.Figure"],
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
            if self.config.scaling_columns is not None:
                logger.info(f"Scaling Columns {self.config.scaling_columns}")
                scaler = ColumnTransformer(
                    [("scalar", StandardScaler(), self.config.scaling_columns)],
                    remainder="passthrough",
                )

                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            self.fit(X_train, y_train)

            y_train_pred = self.model.predict(X_train)
            y_val_pred = self.model.predict(X_val)

            logger.debug("Building training metrics set")
            train_metrics_set = self.config.metrics.clone()
            train_metrics_set.evaluate_all(y_train, y_train_pred)
            all_metrics.add_metrics("train", train_metrics_set)

            logger.debug("Building val metrics set")
            val_metrics_set = self.config.metrics.clone()
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

            fig_data = []
            num_outputs = (
                plotdata[0][0].shape[-1] if len(plotdata[0][0].shape) > 1 else 1
            )

            for j in range(num_outputs):
                # Generate a prediction vs observation plot for all folds
                fig = plt.figure(figsize=(6,6))
                for i, (xdata, ydata) in enumerate(plotdata):
                    x = xdata[:, j] if num_outputs > 1 else xdata
                    y = ydata[:, j] if num_outputs > 1 else ydata
                    plt.scatter(x, y, alpha=0.5, label=f"Fold {i}")

                # Get the current axes, so we can add a line to it
                ax = plt.gca()

                # Get the limits of the current axes
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()

                # Find the range of both axes
                range_all = [np.min([xlim, ylim]), np.max([xlim, ylim])]

                # Plot the diagonal line
                plt.plot(range_all, range_all, "k--")

                plt.xticks(fontsize=11)
                plt.yticks(fontsize=11)
                plt.ylabel("Predicted Values", fontsize=12)
                plt.xlabel("Observed Values", fontsize=12)
                plt.title("Prediction vs Observation Plot", fontsize=15)
                plt.legend()

                if encode_plots_base64:
                    buf = io.BytesIO()
                    plt.savefig(buf, format="png")
                    buf.seek(0)

                    image_64 = base64.b64encode(buf.read())
                    figdata = image_64

                else:
                    figdata = fig

                fig_data.append(figdata)

            return all_metrics, fig_data

        return all_metrics

    def fit_tune(self, X: Any, y: Any | None = None):
        """
        Carry out the actual ray tune operation on the model.
        """
        if self.model is None:
            raise ValueError("no model instance is set -- nothing to tune")

        # Call Setup on the callbacks
        self.trigger_callback("setup", args={"trainer": self})

        def build_model(config) -> None:
            try:
                # Get a clone of the trainer so we can configure the underlying model
                # without affecting other tune instances
                trainer = self.clone()
                trainer.set_custom_config(config)
                logger.debug(f"Metric: {trainer.config.evaluation_metric}")

                trainer.trigger_callback(
                    "on_tune_step_begin", args={"trainer": trainer}
                )

                metrics: NamedAggregatePerformanceMetrics = trainer.cross_validate(
                    X,
                    y,
                    self.config.num_cv_folds,
                    self.config.random_state,
                    False,
                    False,
                    False,
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
                    "val", self.config.evaluation_metric
                )

                logger.debug(f"Avg Metric {self.config.evaluation_metric}:")
                logger.debug(avg_eval_metric)

                results = {}
                for metric_name in metrics.get_available_metrics():
                    results[metric_name] = metrics.get_group_metric_avg(
                        "val", metric_name
                    )
                    results[f"{metric_name}_std"] = metrics.get_group_metric_std(
                        "val", metric_name
                    )

                # If the user used a non standard shortname -- just add the result here
                # there is probably a more efficient method to do this but would require
                # more infrastructure
                if self.config.evaluation_metric not in results:
                    results[
                        self.config.evaluation_metric
                    ] = metrics.get_group_metric_avg(
                        "val", self.config.evaluation_metric
                    )
                    results[
                        f"{self.config.evaluation_metric}_std"
                    ] = metrics.get_group_metric_std(
                        "val", self.config.evaluation_metric
                    )

                tune.report(**results)
            except Exception as e:
                self.trigger_callback("on_tune_step_complete", {"trainer": trainer})
                raise e
            else:
                self.trigger_callback("on_tune_step_complete", {"trainer": trainer})

        if is_notebook():
            logger.info("Starting Notebook Reporter")
            reporter = tune.JupyterNotebookReporter(
                metric_columns=[
                    "loss",
                    "training_iteration",
                    self.config.evaluation_metric,
                ]
            )
        else:
            logger.info("Starting CLI Reporter")
            reporter = tune.CLIReporter(
                metric_columns=[
                    "loss",
                    "training_iteration",
                    self.config.evaluation_metric,
                ]
            )

        hyperopt_search = HyperOptSearch(
            metric=self.config.evaluation_metric,
            mode=self.config.metrics.get_metric_obj(
                self.config.evaluation_metric
            ).optimal_mode(),
        )
        # hyperopt_search = ConcurrencyLimiter(hyperopt_search, max_concurrent=4)

        self.trigger_callback("on_ray_pre_init", {"trainer": self})

        # ray.init(_memory=5000000000, object_store_memory=2000000000, num_cpus=2)
        ray.init()
        analysis = tune.run(
            build_model,
            resources_per_trial={"cpu": 1},
            config=self.config.hyperparameters.get_valid_parameters(
                X.shape, y.shape, (np.min(X), np.max(X)), (np.min(y), np.max(y))
            ),
            search_alg=hyperopt_search,
            progress_reporter=reporter,
            num_samples=self.config.num_hyperopt_samples,
            raise_on_failed_trial=False,
        )

        self.trigger_callback("on_ray_pre_shutdown", {"trainer": self})
        ray.shutdown()

        tdata = analysis.get_best_trial(
            metric=self.config.evaluation_metric,
            mode=self.config.metrics.get_metric_obj(
                self.config.evaluation_metric
            ).optimal_mode(),
        )
        best_params = analysis.get_best_config(
            metric=self.config.evaluation_metric,
            mode=self.config.metrics.get_metric_obj(
                self.config.evaluation_metric
            ).optimal_mode(),
        )

        self.best_params = best_params
        return tdata.last_result, best_params
