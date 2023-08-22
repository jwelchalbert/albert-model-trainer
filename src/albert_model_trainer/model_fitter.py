import copy
import hashlib
import json
import logging
import os
import warnings
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import ray
from albert.internal.utils import hash_string, remove_keys_from_dict
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.base import (BaseEstimator, MultiOutputMixin, RegressorMixin,
                          TransformerMixin)
from sklearn.base import clone as clone_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import (ARDRegression, BayesianRidge, ElasticNet,
                                  HuberRegressor, Lars, Lasso, LassoLars,
                                  LinearRegression, OrthogonalMatchingPursuit,
                                  PassiveAggressiveRegressor, RANSACRegressor,
                                  Ridge, SGDRegressor, TheilSenRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


logger = logging.getLogger("albert.log")

__all__ = ["SklearnRegressorEvaluator"]


def config_to_path(config):
    rconfig = copy.deepcopy(config)
    if "chkpoint_dir" in rconfig:
        del rconfig["chkpoint_dir"]
    if "name" in rconfig:
        del rconfig["name"]
    config_string = (
        json.dumps(rconfig)
        .replace(":", "_")
        .replace("/", "_")
        .replace("{", "")
        .replace("}", "")
        .replace(" ", "")
        .replace('"', "")
        .replace(",", "_")
    )
    return f"{config_string}_model.pkl"


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


class SklearnRegressorEvaluator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        include_adaboost=False,
        n_splits=5,
        random_state=42,
        scaling_columns=None,
        hyper_opt_top_n=False,
        top_n: int | str = 5,
        drop_neg_r2_folds=False,
        progress_callback=None,
        checkpoint_dir=None,
        optimization_metric="r2",
        num_hyperopt_samples=100,
        hyper_tune_complete_callback=None,
    ) -> None:
        self.include_adaboost = include_adaboost
        self.n_splits = n_splits
        self.random_state = random_state
        self.scaling_columns = scaling_columns
        self.hyper_opt_top_n = hyper_opt_top_n
        self.top_n = top_n
        self.drop_neg_r2_folds = drop_neg_r2_folds
        self.progress_callback = progress_callback
        self.optimization_metric = optimization_metric
        self.num_hyperopt_samples = num_hyperopt_samples
        self.hyper_tune_complete_callback = hyper_tune_complete_callback

        if os.path.isabs(checkpoint_dir):
            self.checkpoint_dir = checkpoint_dir
        else:
            self.checkpoint_dir = os.path.join(os.getcwd(), checkpoint_dir)

        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.models = [
            ("LinearRegression", LinearRegression(), {"dummy": tune.uniform(0.0, 1.0)}),
            (
                "Ridge",
                Ridge(random_state=self.random_state),
                {"alpha": tune.loguniform(0.001, 10.0)},
            ),
            (
                "Lasso",
                Lasso(random_state=self.random_state),
                {"alpha": tune.loguniform(0.001, 10.0)},
            ),
            (
                "ElasticNet",
                ElasticNet(random_state=self.random_state),
                {
                    "alpha": tune.loguniform(0.001, 10.0),
                    "l1_ratio": tune.uniform(0.0, 1.0),
                },
            ),
            (
                "Lars",
                Lars(random_state=self.random_state),
                {"n_nonzero_coefs": tune.randint(1, 500)},
            ),
            (
                "LassoLars",
                LassoLars(random_state=self.random_state),
                {"alpha": tune.loguniform(0.001, 10.0)},
            ),
            (
                "OrthogonalMatchingPursuit",
                OrthogonalMatchingPursuit(),
                {"n_nonzero_coefs": tune.randint(1, 500)},
            ),
            (
                "BayesianRidge",
                BayesianRidge(),
                {
                    "alpha_1": tune.loguniform(1e-6, 1e6),
                    "alpha_2": tune.loguniform(1e-6, 1e6),
                    "lambda_1": tune.loguniform(1e-6, 1e6),
                    "lambda_2": tune.loguniform(1e-6, 1e6),
                },
            ),
            # (
            #     "ARDRegression",
            #     ARDRegression(),
            #     {
            #         "alpha_1": tune.loguniform(1e-6, 1e6),
            #         "alpha_2": tune.loguniform(1e-6, 1e6),
            #         "lambda_1": tune.loguniform(1e-6, 1e6),
            #         "lambda_2": tune.loguniform(1e-6, 1e6),
            #     },
            # ),
            (
                "SGDRegressor",
                SGDRegressor(max_iter=1000, tol=1e-3, random_state=self.random_state),
                {
                    "penalty": tune.choice(["l1", "l2", "elasticnet"]),
                    "alpha": tune.loguniform(1e-6, 1e0),
                    "l1_ratio": tune.uniform(0.0, 1.0),
                },
            ),
            # (
            #     "PassiveAggressiveRegressor",
            #     PassiveAggressiveRegressor(
            #         max_iter=1000, tol=1e-3, random_state=self.random_state
            #     ),
            #     {"C": tune.loguniform(1e-6, 1e0)},
            # ),
            # (
            #     "TheilSenRegressor",
            #     TheilSenRegressor(random_state=self.random_state),
            #     {"max_subpopulation": tune.loguniform(1000.0, 10000.0)},
            # ),
            # (
            #     "HuberRegressor",
            #     HuberRegressor(),
            #     {
            #         "epsilon": tune.uniform(1.1, 2.0),
            #         "alpha": tune.loguniform(1e-6, 1e6),
            #     },
            # ),
            (
                "RANSACRegressor",
                RANSACRegressor(random_state=self.random_state),
                {
                    "min_samples": tune.uniform(0.1, 1.0),
                    "residual_threshold": tune.uniform(1.0, 100.0),
                    "max_trials": tune.randint(50, 500),
                },
            ),
            (
                "SVR",
                SVR(),
                {
                    "C": tune.loguniform(0.1, 10.0),
                    "epsilon": tune.uniform(0.01, 0.1),
                    "gamma": tune.loguniform(0.001, 0.1),
                    "kernel": tune.choice(["linear", "poly", "rbf", "sigmoid"]),
                    "degree": tune.choice([2, 3, 4, 5]),
                    "coef0": tune.uniform(0, 1),
                    "shrinking": tune.choice([True, False]),
                    "tol": tune.loguniform(1e-5, 1e-3),
                },
            ),
            (
                "GradientBoostingRegressor",
                GradientBoostingRegressor(random_state=self.random_state),
                {
                    "n_estimators": tune.randint(50, 500),
                    "learning_rate": tune.loguniform(0.01, 0.2),
                    "max_depth": tune.randint(3, 10),
                    "subsample": tune.uniform(0.5, 1),
                    "min_samples_split": tune.randint(2, 10),
                    "min_samples_leaf": tune.randint(1, 10),
                    "max_features": tune.choice([1.0, "sqrt", "log2"]),
                },
            ),
            (
                "RandomForestRegressor",
                RandomForestRegressor(random_state=self.random_state),
                {
                    "n_estimators": tune.randint(50, 500),
                    "max_depth": tune.randint(3, 10),
                    "min_samples_split": tune.randint(2, 10),
                    "min_samples_leaf": tune.randint(1, 10),
                    "criterion": tune.choice(
                        ["absolute_error", "squared_error", "friedman_mse", "poisson"]
                    ),
                    "max_features": tune.choice([1.0, "sqrt", "log2"]),
                    "bootstrap": tune.choice([True, False]),
                    "warm_start": tune.choice([True, False]),
                    "ccp_alpha": tune.uniform(0.0, 0.2),
                },
            ),
            (
                "ExtraTreesRegressor",
                ExtraTreesRegressor(random_state=self.random_state),
                {
                    "n_estimators": tune.randint(50, 500),
                    "max_depth": tune.randint(3, 10),
                    "min_samples_split": tune.randint(2, 10),
                    "min_samples_leaf": tune.randint(1, 10),
                    "max_features": tune.choice(["auto", "sqrt", "log2"]),
                    "bootstrap": tune.choice([True, False]),
                    "warm_start": tune.choice([True, False]),
                    "criterion": tune.choice(
                        ["absolute_error", "squared_error", "friedman_mse", "poisson"]
                    ),
                    "max_leaf_nodes": tune.randint(10, 100),
                    "min_impurity_decrease": tune.uniform(0.0, 0.2),
                    "ccp_alpha": tune.uniform(0.0, 0.2),
                },
            ),
            (
                "KNeighborsRegressor",
                KNeighborsRegressor(),
                {
                    "n_neighbors": tune.randint(1, 10),
                    "weights": tune.choice(["uniform", "distance"]),
                    "p": tune.choice([1, 2]),
                    "algorithm": tune.choice(["auto", "ball_tree", "kd_tree", "brute"]),
                    "leaf_size": tune.randint(1, 50),
                    "metric": tune.choice(
                        ["euclidean", "manhattan", "chebyshev", "minkowski"]
                    ),
                },
            ),
            (
                "DecisionTreeRegressor",
                DecisionTreeRegressor(random_state=self.random_state),
                {
                    "max_depth": tune.randint(1, 10),
                    "min_samples_split": tune.randint(2, 10),
                    "min_samples_leaf": tune.randint(1, 10),
                    "criterion": tune.choice(
                        ["squared_error", "absolute_error", "poisson", "friedman_mse"]
                    ),
                    "splitter": tune.choice(["best", "random"]),
                    "max_features": tune.choice(["auto", "sqrt", "log2"]),
                    "max_leaf_nodes": tune.randint(10, 100),
                    "min_impurity_decrease": tune.uniform(0.0, 0.2),
                    "ccp_alpha": tune.uniform(0.0, 0.2),
                },
            ),
            (
                "MLPRegressor",
                MLPRegressor(max_iter=1000, random_state=self.random_state),
                {
                    "hidden_layer_sizes": tune.choice([(50,), (100,), (200,)]),
                    "activation": tune.choice(["relu", "tanh", "logistic"]),
                    "alpha": tune.loguniform(0.0001, 0.1),
                    "solver": tune.choice(["lbfgs", "sgd", "adam"]),
                    "learning_rate": tune.choice(
                        ["constant", "invscaling", "adaptive"]
                    ),
                    "max_iter": tune.randint(200, 2000),
                    "tol": tune.loguniform(0.0001, 0.1),
                    "momentum": tune.uniform(0.1, 0.9),
                    "nesterovs_momentum": tune.choice([True, False]),
                    "beta_1": tune.uniform(0.1, 0.9),
                    "beta_2": tune.uniform(0.1, 0.999),
                    "epsilon": tune.loguniform(1e-9, 1e-7),
                    "n_iter_no_change": tune.randint(5, 50),
                    "max_fun": tune.randint(10000, 50000),
                },
            ),
        ]

        self.no_adaboost_names = [
            "OrthogonalMatchingPursuit",
            "BayesianRidge",
            "ARDRegression",
            "PassiveAggressiveRegressor",
            "TheilSenRegressor",
            "HuberRegressor",
            "RANSACRegressor",
            "SVR",
        ]

        if self.include_adaboost:
            adaboost_models = []
            for name, model, hps in tqdm(
                self.models, desc="Setting up Adaboost Models"
            ):
                if name in self.no_adaboost_names:
                    continue
                boost_model = clone_model(model)
                boost_name = f"{name}_Adaboost"
                adaboost_models.append(
                    (
                        boost_name,
                        AdaBoostRegressor(
                            boost_model,
                            n_estimators=100,
                            random_state=self.random_state,
                        ),
                        hps,
                    )
                )
            self.models.extend(adaboost_models)

        if (self.top_n in [-1, "all"]) or (self.top_n > len(self.models)):
            logger.info("Optimizing all models")
            print("Optimizing all models")
            self.top_n = len(self.models)

    def fit(self, X: Any, y: Any | None = None):
        assert y is not None, "cannot regress without a target"
        assert len(y.shape) <= 2, "only tensors of shape [B, E] are allowed as outputs"

        self.multi_output = False
        if (len(y.shape) > 1) and (y.shape[-1] > 1):
            self.multi_output = True

        model_results = []
        self.model_metrics = []
        total_models = len(self.models)
        tc = 0

        if self.hyper_opt_top_n is False or (
            (self.hyper_opt_top_n is True) and (self.top_n != len(self.models))
        ):
            for name, model, _ in tqdm(self.models, desc="Fitting Regressor Model Zoo"):
                if self.multi_output:
                    model = MultiOutputRegressor(model)

                metrics = self.cross_validate(
                    model, X, y, self.n_splits, self.random_state
                )
                self.model_metrics.append((name, metrics))

                avg_train_rmse = np.array(metrics["train"]["rmse"]).mean()
                avg_val_rmse = np.array(metrics["val"]["rmse"]).mean()
                avg_train_mae = np.array(metrics["train", "mae"]).mean()
                avg_val_mae = np.array(metrics["val"]["mae"]).mean()
                avg_train_r2 = np.array(metrics["train"]["r2"]).mean()
                avg_val_r2 = np.array(metrics["val"]["r2"])
                if self.drop_neg_r2_folds:
                    mask = avg_val_r2 >= 0.0
                    avg_val_r2 = avg_val_r2[mask]
                    std_val_r2 = np.array(metrics["val"]["r2"])[mask]
                    if len(avg_val_r2) > 0:
                        avg_val_r2 = avg_val_r2.mean()
                        std_val_r2 = std_val_r2.std()
                    else:
                        avg_val_r2 = 0
                        std_val_r2 = np.inf
                else:
                    avg_val_r2 = avg_val_r2.mean()
                    std_val_r2 = np.array(metrics["val"]["r2"]).std()

                model_results.append(
                    (
                        name,
                        avg_train_rmse,
                        avg_val_rmse,
                        avg_train_r2,
                        avg_val_r2,
                        std_val_r2,
                        avg_train_mae,
                        avg_val_mae,
                    )
                )
                if self.progress_callback is not None:
                    tc += 1
                    self.progress_callback(tc, "base")

            model_results.sort(key=lambda x: x[4], reverse=True)
        else:
            # Hyperopt was requested on all -- fill in the model_results with dummy info
            # so that the logic below can operate correctly
            model_results = [(x[0], 0, 0, 0, 0, 0) for x in self.models]

            # If there is a progress callback registered for the base model build
            # the indicate that all base models are done since we aren't going to do this
            if self.progress_callback is not None:
                self.progress_callback(len(self.models), "base")

        if self.hyper_opt_top_n:
            tc = 0
            self.best_params = []
            top_models = model_results[: self.top_n]
            mnames, mmodels, mparams = zip(*self.models, strict=False)
            model_params = dict(zip(mnames, zip(mmodels, mparams)), strict=False)
            for name, _, _, _, _, _ in top_models:
                skip_model = False

                if name in ["Lars"]:
                    # Hack: don't process Lars models on
                    # dfs that have less than 3 features
                    if len(X.shape) < 2:
                        logger.info("Fewer than 3 features, skipping LARS")
                        skip_model = True
                    elif X.shape[-1] < 3:
                        logger.info("Fewer than 3 features, skipping LARS")
                        skip_model = True

                if not skip_model:
                    model, params = model_params[name]
                    params["name"] = name
                    if self.checkpoint_dir is not None:
                        params["chkpoint_dir"] = self.checkpoint_dir

                    tdata, best_params = self.hyperparameter_optimization(
                        model, params, X, y
                    )

                    self.best_params.append((name, tdata, best_params))

                # If there is a progress callback then call it
                if self.progress_callback is not None:
                    tc += 1
                    self.progress_callback(tc, "hyperopt")

            return self.best_params
        else:
            self.model_results = model_results
            return model_results

    def get_top_model_pipeline(self, nth_model=0, return_name_score=False):
        top_model, nstuple = self.get_nth_top_model(nth_model, True)
        # Create the scalar if we were told to scale certain columns
        if self.scaling_columns is not None:
            # logger.info(f"Scaling Columns {self.scaling_columns}")
            scaler = ColumnTransformer(
                [("scalar", StandardScaler(), self.scaling_columns)],
                remainder="passthrough",
            )
            pipe = Pipeline([("scaler", scaler), ("model", top_model)])
        else:
            pipe = Pipeline([("model", top_model)])

        if return_name_score:
            return pipe, nstuple
        else:
            return pipe

    def get_top_model(self, return_name_score=False):
        return self.get_nth_top_model(0, return_name_score)

    def get_nth_top_model(self, n: int, return_name_score=False):
        assert n < len(
            self.best_params
        ), "n must be less than the number of tuned models"

        best_sorted = sorted(
            list(map(lambda x: (x[0], x[1]["r2"], x[2]), self.best_params)),
            key=lambda x: x[1],
            reverse=True,
        )
        name, score, params = best_sorted[n]
        best_model = list(filter(lambda x: x[0] == name, self.models))

        best_model_inst = clone_model(best_model[0][1])
        for param, value in remove_keys_from_dict(
            params, ["name", "chkpoint_dir"]
        ).items():
            if isinstance(best_model_inst, AdaBoostRegressor):
                setattr(best_model_inst.estimator, param, value)
            else:
                setattr(best_model_inst, param, value)

        logger.info(f"Top [{n}] model with a score of [{score}] -- [{name}]")
        if self.multi_output:
            logger.debug("multi output flag set -- returning multi output regressor")
            best_model_inst = MultiOutputRegressor(best_model_inst)

        if return_name_score:
            return best_model_inst, (name, score)
        else:
            return best_model_inst

    def cross_validate(
        self,
        model,
        X,
        y,
        n_splits=5,
        random_state=42,
        display_fold_progress=False,
        generate_pred_obs_plots=False,
        encode_plots_base64=False,
    ):
        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        plotdata: list[Tuple[np.ndarray, np.ndarray]] = []

        metrics = {
            "train": {"r2": [], "mae": [], "mse": [], "rmse": []},
            "val": {"r2": [], "mae": [], "mse": [], "rmse": []},
        }

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

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            metrics["train"]["r2"].append(r2_score(y_train, y_train_pred))
            metrics["train"]["mae"].append(mean_absolute_error(y_train, y_train_pred))
            metrics["train"]["mse"].append(mean_squared_error(y_train, y_train_pred))
            metrics["train"]["rmse"].append(
                mean_squared_error(y_train, y_train_pred, squared=False)
            )

            metrics["val"]["r2"].append(r2_score(y_val, y_val_pred))
            metrics["val"]["mae"].append(mean_absolute_error(y_val, y_val_pred))
            metrics["val"]["mse"].append(mean_squared_error(y_val, y_val_pred))
            metrics["val"]["rmse"].append(
                mean_squared_error(y_val, y_val_pred, squared=False)
            )

            if generate_pred_obs_plots:
                # Store the data for generating the plot later
                plotdata.append((y_val, y_val_pred))

        if generate_pred_obs_plots:
            import matplotlib

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
                metrics["plots"] = image_64

            else:
                metrics["plots"] = fig

        return metrics

    def hyperparameter_optimization(self, model, params, X, y):
        multi_output = False
        if (len(y.shape) > 1) and (y.shape[-1] > 1):
            multi_output = True

        def objective(config):
            mm = clone_model(model)
            checkpoint_dir = None
            for param, val in config.items():
                # If we are wrapping the estimator with adaboost, then set the attributes
                # on the underlying estimator

                if param == "chkpoint_dir":
                    checkpoint_dir = val
                    continue

                if param == "name":
                    continue

                if isinstance(mm, AdaBoostRegressor):
                    setattr(mm.estimator, param, val)
                else:
                    setattr(mm, param, val)

            if multi_output:
                mm = MultiOutputRegressor(mm)

            # Run the cross validation using the given parameters
            metrics = self.cross_validate(mm, X, y, random_state=self.random_state)
            avg_val_r2 = np.array(metrics["val"]["r2"])
            if self.drop_neg_r2_folds:
                mask = avg_val_r2 >= 0.0
                avg_val_r2 = avg_val_r2[mask]
                std_val_r2 = np.array(metrics["val"]["r2"])[mask]
                if len(avg_val_r2) > 0:
                    avg_val_r2 = avg_val_r2.mean()
                    std_val_r2 = std_val_r2.std()
                else:
                    avg_val_r2 = 0
                    std_val_r2 = np.inf
            else:
                avg_val_r2 = avg_val_r2.mean()
                std_val_r2 = np.array(metrics["val"]["r2"]).std()

            r2vals = np.array(avg_val_r2)
            tune.report(r2=r2vals.mean())
            if self.hyper_tune_complete_callback is not None:
                self.hyper_tune_complete_callback("step")

            # if checkpoint_dir is not None:
            #     opath = os.path.join(
            #         checkpoint_dir,
            #         config["name"],
            #     )
            #     if not os.path.isdir(opath):
            #         os.makedirs(opath)

            #     # Save the model with a filename that includes the parameters
            #     joblib.dump(
            #         mm,
            #         os.path.join(opath, hash_string(config_to_path(config))),
            #     )

            # model.fit(X_train, y_train)
            # preds = model.predict(X_val)
            # error = r2_score(y_val, preds)
            # tune.report(r2=error)

        if is_notebook():
            logger.info("Starting Notebook Reporter")
            reporter = tune.JupyterNotebookReporter(
                metric_columns=["loss", "training_iteration", "r2"]
            )
        else:
            logger.info("Starting CLI Reporter")
            reporter = tune.CLIReporter(
                metric_columns=["loss", "training_iteration", "r2"]
            )

        for param, val in params.items():
            # Check that sparsity bound parameters do
            # not exceed the number of dimensions in our feature space
            if param == "n_nonzero_coefs":
                # first check that we have more than 1 input feature
                if len(X.shape) < 2:
                    val.upper = 1  # We have a single input dimension we have to have atoms == dim(X,1)
                else:
                    if val.upper > X.shape[-1]:
                        val.upper = X.shape[-1]
                params[param] = val

                print("NNC Val Range:", val.upper, val.lower)

        hyperopt_search = HyperOptSearch(metric="r2", mode="max")
        hyperopt_search = ConcurrencyLimiter(hyperopt_search, max_concurrent=4)

        def trial_name_string(trial):
            return str(trial.trial_id)

        print("Ray Started")
        print(params)
        import time

        if self.hyper_tune_complete_callback is not None:
            if isinstance(model, AdaBoostRegressor):
                model_name = type(model.estimator).__name__
            else:
                model_name = type(model).__name__

            self.hyper_tune_complete_callback(f"model:{model_name}")

        time.sleep(2)
        ray.init(_memory=5000000000, object_store_memory=2000000000, num_cpus=2)
        analysis = tune.run(
            objective,
            resources_per_trial={"cpu": 1},
            config=params,
            search_alg=hyperopt_search,
            progress_reporter=reporter,
            num_samples=self.num_hyperopt_samples,
            raise_on_failed_trial=False,
            # trial_name_creator=trial_name_string,
        )
        ray.shutdown()
        print("Ray Shutdown")

        # analysis = tune.run(
        #     objective,
        #     config=params,
        #     metric="r2",
        #     mode="max",
        #     progress_reporter=reporter,
        #     num_samples=20,
        #     raise_on_failed_trial=False,
        # )
        tdata = analysis.get_best_trial(metric="r2", mode="max")
        best_params = analysis.get_best_config(metric="r2", mode="max")

        return tdata.last_result, best_params

    def predict(self, X):
        return None
        # self.check_is_fitted()
        # y_pred = []
        # for estimator in self.estimators_:
        #     y_pred.append(estimator.predict(X))
        # return np.column_stack(y_pred)

    def check_is_fitted(self):
        if not hasattr(self, "estimators_"):
            raise ValueError(
                "The model is not fitted yet. Please call 'fit' with appropriate arguments before using this estimator."
            )
