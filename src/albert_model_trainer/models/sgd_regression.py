from typing import Any

from ray import tune
from sklearn.linear_model import SGDRegressor

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (
    ModelConfigurationBase,
    validate_config_type,
)


class SGDRegressorHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_PENALTY = tune.choice(["l1", "l2", "elasticnet"])
    DEFAULT_ALPHA = tune.loguniform(1e-6, 1e0)
    DEFAULT_L1_RATIO = tune.uniform(0.0, 1.0)

    def __init__(self, **kwargs) -> None:
        self._parameters = {
            "penalty": kwargs.get("penalty", self.DEFAULT_PENALTY),
            "alpha": kwargs.get("alpha", self.DEFAULT_ALPHA),
            "l1_ratio": kwargs.get("l1_ratio", self.DEFAULT_L1_RATIO),
        }


class SGDRegressorTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(SGDRegressorHyperparameterSet())
        else:
            validate_config_type(config, SGDRegressorHyperparameterSet)

        super().__init__(config)
        self.model = SGDRegressor(
            max_iter=1000, tol=1e-3, random_state=config.random_state
        )

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
