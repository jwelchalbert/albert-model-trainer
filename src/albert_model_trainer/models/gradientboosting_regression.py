from typing import Any

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from ray import tune
from sklearn.ensemble import (
    GradientBoostingRegressor,
)
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (
    ModelConfigurationBase,
    validate_config_type,
)


class GradientBoostingRegressorHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_N_ESTIMATORS = tune.randint(50, 500)
    DEFAULT_LEARNING_RATE = tune.loguniform(0.01, 0.2)
    DEFAULT_MAX_DEPTH = tune.randint(3, 10)
    DEFAULT_SUBSAMPLE = tune.uniform(0.5, 1)
    DEFAULT_MIN_SAMPLES_SPLIT = tune.randint(2, 10)
    DEFAULT_MIN_SAMPLES_LEAF = tune.randint(1, 10)
    DEFAULT_MAX_FEATURES = tune.choice([1.0, "sqrt", "log2"])

    def __init__(self, **kwargs) -> None:
        self.parameters = {
            "n_estimators": kwargs.get("n_estimators", self.DEFAULT_N_ESTIMATORS),
            "learning_rate": kwargs.get("learning_rate", self.DEFAULT_LEARNING_RATE),
            "max_depth": kwargs.get("max_depth", self.DEFAULT_MAX_DEPTH),
            "subsample": kwargs.get("subsample", self.DEFAULT_SUBSAMPLE),
            "min_samples_split": kwargs.get(
                "min_samples_split", self.DEFAULT_MIN_SAMPLES_SPLIT
            ),
            "min_samples_leaf": kwargs.get(
                "min_samples_leaf", self.DEFAULT_MIN_SAMPLES_LEAF
            ),
            "max_features": kwargs.get("max_features", self.DEFAULT_MAX_FEATURES),
        }


class GradientBoostingRegressorTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(
                GradientBoostingRegressorHyperparameterSet()
            )
        else:
            validate_config_type(config, GradientBoostingRegressorHyperparameterSet)

        super().__init__(config)
        self.model = GradientBoostingRegressor(random_state=config.random_state)

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
