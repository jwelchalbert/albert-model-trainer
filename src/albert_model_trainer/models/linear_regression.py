from typing import Any

from ray import tune
from sklearn.linear_model import LinearRegression

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (
    ModelConfigurationBase,
    validate_config_type,
)


class LinearRegressionHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_DUMMY = tune.loguniform(0.001, 10.0)

    def __init__(self, **kwargs) -> None:
        self._parameters = {
            "dummy": kwargs.get("dummy", self.DEFAULT_DUMMY),
        }


class LinearRegressionTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(LinearRegressionHyperparameterSet())
        else:
            validate_config_type(config, LinearRegressionHyperparameterSet)

        super().__init__(config)
        self.model = LinearRegression()

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
