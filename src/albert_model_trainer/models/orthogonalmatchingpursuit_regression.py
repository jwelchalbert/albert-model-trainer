from typing import Any, Tuple

from ray import tune
from sklearn.linear_model import OrthogonalMatchingPursuit

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (
    ModelConfigurationBase,
    validate_config_type,
)
from loguru import logger


class OrthogonalMatchingPursuitHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_N_NONZERO_COEFS = tune.randint(1, 500)

    def __init__(self, **kwargs) -> None:
        self._parameters = {
            "n_nonzero_coefs": kwargs.get(
                "n_nonzero_coefs", self.DEFAULT_N_NONZERO_COEFS
            ),
        }

    def get_valid_parameters(
        self,
        input_shape: Tuple,
        output_shape: Tuple,
        intput_ranges: Tuple,
        output_ranges: Tuple,
    ) -> dict[str, Any]:
        new_params = self.parameters.copy()
        for param, val in self.parameters.items():
            if param == "n_nonzero_coefs":
                if len(input_shape) < 2:
                    val.upper = 1
                else:
                    if val.upper > input_shape[-1]:
                        val.upper = input_shape[-1]
                new_params[param] = val
                logger.info("NNC Val Range:", val.upper, val.lower)
        return new_params


class OrthogonalMatchingPursuitTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(
                OrthogonalMatchingPursuitHyperparameterSet()
            )
        else:
            validate_config_type(config, OrthogonalMatchingPursuitHyperparameterSet)

        super().__init__(config)
        self.model = OrthogonalMatchingPursuit()

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
