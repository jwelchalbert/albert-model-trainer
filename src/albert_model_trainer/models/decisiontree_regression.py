from typing import Any, Tuple

from ray import tune
from sklearn.tree import DecisionTreeRegressor

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (
    ModelConfigurationBase,
    validate_config_type,
)

from loguru import logger


class DecisionTreeRegressorHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_MAX_DEPTH = tune.randint(1, 10)
    DEFAULT_MIN_SAMPLES_SPLIT = tune.randint(2, 10)
    DEFAULT_MIN_SAMPLES_LEAF = tune.randint(1, 10)
    DEFAULT_CRITERION = tune.choice(
        ["squared_error", "absolute_error", "poisson", "friedman_mse"]
    )
    DEFAULT_SPLITTER = tune.choice(["best", "random"])
    DEFAULT_MAX_FEATURES = tune.choice(["sqrt", "log2"])
    DEFAULT_MAX_LEAF_NODES = tune.randint(10, 100)
    DEFAULT_MIN_IMPURITY_DECREASE = tune.uniform(0.0, 0.2)
    DEFAULT_CCP_ALPHA = tune.uniform(0.0, 0.2)

    def __init__(self, **kwargs) -> None:
        self._parameters = {
            "max_depth": kwargs.get("max_depth", self.DEFAULT_MAX_DEPTH),
            "min_samples_split": kwargs.get(
                "min_samples_split", self.DEFAULT_MIN_SAMPLES_SPLIT
            ),
            "min_samples_leaf": kwargs.get(
                "min_samples_leaf", self.DEFAULT_MIN_SAMPLES_LEAF
            ),
            "criterion": kwargs.get("criterion", self.DEFAULT_CRITERION),
            "splitter": kwargs.get("splitter", self.DEFAULT_SPLITTER),
            "max_features": kwargs.get("max_features", self.DEFAULT_MAX_FEATURES),
            "max_leaf_nodes": kwargs.get("max_leaf_nodes", self.DEFAULT_MAX_LEAF_NODES),
            "min_impurity_decrease": kwargs.get(
                "min_impurity_decrease", self.DEFAULT_MIN_IMPURITY_DECREASE
            ),
            "ccp_alpha": kwargs.get("ccp_alpha", self.DEFAULT_CCP_ALPHA),
        }

    def get_valid_parameters(
        self,
        input_shape: Tuple,
        output_shape: Tuple,
        intput_ranges: Tuple,
        output_ranges: Tuple,
    ) -> dict[str, Any]:
        # Poisson criterion doesn't support negative y values
        valid_params = self.parameters.copy()
        if output_ranges[0] < 0 or output_ranges[1] < 0:
            if "poisson" in valid_params["criterion"]:
                criterions = list(valid_params["criterion"])
                if "poisson" in criterions:
                    criterions.remove("poisson")
                    valid_params["criterion"] = tune.choice(criterions)
                    logger.debug(
                        "Removed Poisson from criterion since output contains negative values"
                    )

        return valid_params


class DecisionTreeRegressionTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(DecisionTreeRegressorHyperparameterSet())
        else:
            validate_config_type(config, DecisionTreeRegressorHyperparameterSet)

        super().__init__(config)
        self.model = DecisionTreeRegressor(random_state=config.random_state)

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
