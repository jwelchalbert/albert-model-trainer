from typing import Any

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from ray import tune
from sklearn.linear_model import (
    RANSACRegressor,
)
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (
    ModelConfigurationBase,
    validate_config_type,
)


class RANSACRegressorHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_MIN_SAMPLES = tune.uniform(0.1, 1.0)
    DEFAULT_RESIDUAL_THRESHOLD = tune.uniform(1.0, 100.0)
    DEFAULT_MAX_TRIALS = tune.randint(50, 500)

    def __init__(self, **kwargs) -> None:
        self.parameters = {
            "min_samples": kwargs.get("min_samples", self.DEFAULT_MIN_SAMPLES),
            "residual_threshold": kwargs.get(
                "residual_threshold", self.DEFAULT_RESIDUAL_THRESHOLD
            ),
            "max_trials": kwargs.get("max_trials", self.DEFAULT_MAX_TRIALS),
        }


class RANSACRegressorTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(RANSACRegressorHyperparameterSet())
        else:
            validate_config_type(config, RANSACRegressorHyperparameterSet)

        super().__init__(config)
        self.model = RANSACRegressor(random_state=config.random_state)

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
