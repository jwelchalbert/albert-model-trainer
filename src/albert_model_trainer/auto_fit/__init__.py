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
        pass