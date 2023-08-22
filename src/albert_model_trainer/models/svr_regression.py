from typing import Any

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from ray import tune
from sklearn.svm import SVR
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (
    ModelConfigurationBase,
    validate_config_type,
)


class SVRHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_C = tune.loguniform(0.1, 10.0)
    DEFAULT_EPSILON = tune.uniform(0.01, 0.1)
    DEFAULT_GAMMA = tune.loguniform(0.001, 0.1)
    DEFAULT_KERNEL = tune.choice(["linear", "poly", "rbf", "sigmoid"])
    DEFAULT_DEGREE = tune.choice([2, 3, 4, 5])
    DEFAULT_COEF0 = tune.uniform(0, 1)
    DEFAULT_SHRINKING = tune.choice([True, False])
    DEFAULT_TOL = tune.loguniform(1e-5, 1e-3)

    def __init__(self, **kwargs) -> None:
        self.parameters = {
            "C": kwargs.get("C", self.DEFAULT_C),
            "epsilon": kwargs.get("epsilon", self.DEFAULT_EPSILON),
            "gamma": kwargs.get("gamma", self.DEFAULT_GAMMA),
            "kernel": kwargs.get("kernel", self.DEFAULT_KERNEL),
            "degree": kwargs.get("degree", self.DEFAULT_DEGREE),
            "coef0": kwargs.get("coef0", self.DEFAULT_COEF0),
            "shrinking": kwargs.get("shrinking", self.DEFAULT_SHRINKING),
            "tol": kwargs.get("tol", self.DEFAULT_TOL),
        }


class SVRTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(SVRHyperparameterSet())
        else:
            validate_config_type(config, SVRHyperparameterSet)

        super().__init__(config)
        self.model = SVR()

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
