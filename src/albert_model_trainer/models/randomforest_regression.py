from typing import Any

from ray import tune
from sklearn.ensemble import RandomForestRegressor

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (ModelConfigurationBase,
                                                    validate_config_type)


class RandomForestRegressorHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_N_ESTIMATORS = tune.randint(50, 500)
    DEFAULT_MAX_DEPTH = tune.randint(3, 10)
    DEFAULT_MIN_SAMPLES_SPLIT = tune.randint(2, 10)
    DEFAULT_MIN_SAMPLES_LEAF = tune.randint(1, 10)
    DEFAULT_CRITERION = tune.choice(
        ["absolute_error", "squared_error", "friedman_mse", "poisson"]
    )
    DEFAULT_MAX_FEATURES = tune.choice([1.0, "sqrt", "log2"])
    DEFAULT_BOOTSTRAP = tune.choice([True, False])
    DEFAULT_WARM_START = tune.choice([True, False])
    DEFAULT_CCP_ALPHA = tune.uniform(0.0, 0.2)

    def __init__(self, **kwargs) -> None:
        self.parameters = {
            "n_estimators": kwargs.get("n_estimators", self.DEFAULT_N_ESTIMATORS),
            "max_depth": kwargs.get("max_depth", self.DEFAULT_MAX_DEPTH),
            "min_samples_split": kwargs.get(
                "min_samples_split", self.DEFAULT_MIN_SAMPLES_SPLIT
            ),
            "min_samples_leaf": kwargs.get(
                "min_samples_leaf", self.DEFAULT_MIN_SAMPLES_LEAF
            ),
            "criterion": kwargs.get("criterion", self.DEFAULT_CRITERION),
            "max_features": kwargs.get("max_features", self.DEFAULT_MAX_FEATURES),
            "bootstrap": kwargs.get("bootstrap", self.DEFAULT_BOOTSTRAP),
            "warm_start": kwargs.get("warm_start", self.DEFAULT_WARM_START),
            "ccp_alpha": kwargs.get("ccp_alpha", self.DEFAULT_CCP_ALPHA),
        }


class RandomForestRegressorTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(RandomForestRegressorHyperparameterSet())
        else:
            validate_config_type(config, RandomForestRegressorHyperparameterSet)

        super().__init__(config)
        self.model = RandomForestRegressor(random_state=config.random_state)

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
