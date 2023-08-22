from typing import Any

from ray import tune
from sklearn.linear_model import Ridge

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (ModelConfigurationBase,
                                                    validate_config_type)


# (
# "Ridge",
# Ridge(random_state=self.random_state),
# {"alpha": tune.loguniform(0.001, 10.0)},
# ),
class RidgeRegressionHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_ALPHA = tune.loguniform(0.001, 10.0)

    def __init__(self, **kwargs) -> None:
        self.parameters = {"alpha": kwargs.get("alpha", self.DEFAULT_ALPHA)}


class RidgeRegressionTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(RidgeRegressionHyperparameterSet())
        else:
            validate_config_type(config, RidgeRegressionHyperparameterSet)

        super().__init__(config)
        self.model = Ridge(random_state=config.random_state)

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
