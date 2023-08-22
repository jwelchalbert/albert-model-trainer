from typing import Any

from ray import tune
from sklearn.neighbors import KNeighborsRegressor

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (ModelConfigurationBase,
                                                    validate_config_type)


class KNeighborsRegressorHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_N_NEIGHBORS = tune.randint(1, 10)
    DEFAULT_WEIGHTS = tune.choice(["uniform", "distance"])
    DEFAULT_P = tune.choice([1, 2])
    DEFAULT_ALGORITHM = tune.choice(["auto", "ball_tree", "kd_tree", "brute"])
    DEFAULT_LEAF_SIZE = tune.randint(1, 50)
    DEFAULT_METRIC = tune.choice(["euclidean", "manhattan", "chebyshev", "minkowski"])

    def __init__(self, **kwargs) -> None:
        self.parameters = {
            "n_neighbors": kwargs.get("n_neighbors", self.DEFAULT_N_NEIGHBORS),
            "weights": kwargs.get("weights", self.DEFAULT_WEIGHTS),
            "p": kwargs.get("p", self.DEFAULT_P),
            "algorithm": kwargs.get("algorithm", self.DEFAULT_ALGORITHM),
            "leaf_size": kwargs.get("leaf_size", self.DEFAULT_LEAF_SIZE),
            "metric": kwargs.get("metric", self.DEFAULT_METRIC),
        }


class KNeighborsRegressorTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(KNeighborsRegressorHyperparameterSet())
        else:
            validate_config_type(config, KNeighborsRegressorHyperparameterSet)

        super().__init__(config)
        self.model = KNeighborsRegressor()

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
