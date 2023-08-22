from typing import Any

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from ray import tune
from sklearn.linear_model import (
    LassoLars,
)
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (
    ModelConfigurationBase,
    validate_config_type,
)


class LassoLarsRegressionHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_ALPHA = tune.loguniform(0.001, 10.0)

    def __init__(self, **kwargs) -> None:
        self.parameters = {
            "alpha": kwargs.get("alpha", self.DEFAULT_ALPHA),
        }


class LassoLarsRegressionTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(LassoLarsRegressionHyperparameterSet())
        else:
            validate_config_type(config, LassoLarsRegressionHyperparameterSet)

        super().__init__(config)
        self.model = LassoLars(random_state=config.random_state)

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
