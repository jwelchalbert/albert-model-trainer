from typing import Any

from ray import tune
from sklearn.linear_model import BayesianRidge

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (
    ModelConfigurationBase,
    validate_config_type,
)


class BayesianRidgeRegressionHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_ALPHA_1 = tune.loguniform(1e-6, 1e6)
    DEFAULT_ALPHA_2 = tune.loguniform(1e-6, 1e6)
    DEFAULT_LAMBDA_1 = tune.loguniform(1e-6, 1e6)
    DEFAULT_LAMBDA_2 = tune.loguniform(1e-6, 1e6)

    def __init__(self, **kwargs) -> None:
        self._parameters = {
            "alpha_1": kwargs.get("alpha_1", self.DEFAULT_ALPHA_1),
            "alpha_2": kwargs.get("alpha_2", self.DEFAULT_ALPHA_2),
            "lambda_1": kwargs.get("lambda_1", self.DEFAULT_LAMBDA_1),
            "lambda_2": kwargs.get("lambda_2", self.DEFAULT_LAMBDA_2),
        }


class BayesianRidgeRegressionTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(BayesianRidgeRegressionHyperparameterSet())
        else:
            validate_config_type(config, BayesianRidgeRegressionHyperparameterSet)

        super().__init__(config)
        self.model = BayesianRidge()

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
