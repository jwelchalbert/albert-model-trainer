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
from albert_model_trainer.base.callback import Callback
from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
import ray
from ray import tune
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.base import (
    BaseEstimator,
    MultiOutputMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.base import clone as clone_model
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ARDRegression,
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    OrthogonalMatchingPursuit,
    PassiveAggressiveRegressor,
    RANSACRegressor,
    Ridge,
    SGDRegressor,
    TheilSenRegressor,
)
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
from albert_model_trainer.base.metrics import Metric, PerformanceMetrics
from albert_model_trainer.base.model import ModelTrainer


# (
# "Ridge",
# Ridge(random_state=self.random_state),
# {"alpha": tune.loguniform(0.001, 10.0)},
# ),
class RidgeRegressionHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_ALPHA = tune.loguniform(0.001, 10.0)

    def __init__(self, **kwargs) -> None:
        self.parameters = {"alpha": kwargs.get("alpha", self.DEFAULT_ALPHA)}


class RidgeRegressionTrainer(ModelTrainer):
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
        if hyperparameters is None:
            hyperparameters = RidgeRegressionHyperparameterSet()

        super().__init__(
            hyperparameters,
            num_cv_folds,
            evaluation_metric,
            random_state,
            metrics,
            callbacks,
            scaling_columns,
            num_hyperopt_samples,
        )

        self.model = Ridge(random_state=random_state)

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
