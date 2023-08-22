from typing import Any

from ray import tune
from sklearn.linear_model import Lars

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (ModelConfigurationBase,
                                                    validate_config_type)


class LarsRegressionHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_N_NONZERO_COEFS = tune.randint(1, 500)

    def __init__(self, **kwargs) -> None:
        self.parameters = {
            "n_nonzero_coefs": kwargs.get(
                "n_nonzero_coefs", self.DEFAULT_N_NONZERO_COEFS
            ),
        }


class LarsRegressionTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(LarsRegressionHyperparameterSet())
        else:
            validate_config_type(config, LarsRegressionHyperparameterSet)

        super().__init__(config)
        self.model = Lars(random_state=config.random_state)

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
