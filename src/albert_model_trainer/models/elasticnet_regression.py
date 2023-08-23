from typing import Any

from ray import tune
from sklearn.linear_model import ElasticNet

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (
    ModelConfigurationBase,
    validate_config_type,
)


class ElasticNetRegressionHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_ALPHA = tune.loguniform(0.001, 10.0)
    DEFAULT_L1_RATIO = tune.uniform(0.0, 1.0)

    def __init__(self, **kwargs) -> None:
        self._parameters = {
            "alpha": kwargs.get("alpha", self.DEFAULT_ALPHA),
            "l1_ratio": kwargs.get("l1_ratio", self.DEFAULT_L1_RATIO),
        }


class ElasticNetRegressionTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(ElasticNetRegressionHyperparameterSet())
        else:
            validate_config_type(config, ElasticNetRegressionHyperparameterSet)

        super().__init__(config)
        self.model = ElasticNet(random_state=config.random_state)

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
