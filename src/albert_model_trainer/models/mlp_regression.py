from typing import Any

from ray import tune
from sklearn.neural_network import MLPRegressor

from albert_model_trainer.base.hyperparameter import HyperParameterTuneSet
from albert_model_trainer.base.model import ModelTrainer
from albert_model_trainer.base.model_config import (ModelConfigurationBase,
                                                    validate_config_type)


class MLPRegressorHyperparameterSet(HyperParameterTuneSet):
    DEFAULT_HIDDEN_LAYER_SIZES = tune.choice([(50,), (100,), (200,)])
    DEFAULT_ACTIVATION = tune.choice(["relu", "tanh", "logistic"])
    DEFAULT_ALPHA = tune.loguniform(0.0001, 0.1)
    DEFAULT_SOLVER = tune.choice(["lbfgs", "sgd", "adam"])
    DEFAULT_LEARNING_RATE = tune.choice(["constant", "invscaling", "adaptive"])
    DEFAULT_MAX_ITER = tune.randint(200, 2000)
    DEFAULT_TOL = tune.loguniform(0.0001, 0.1)
    DEFAULT_MOMENTUM = tune.uniform(0.1, 0.9)
    DEFAULT_NESTEROVS_MOMENTUM = tune.choice([True, False])
    DEFAULT_BETA_1 = tune.uniform(0.1, 0.9)
    DEFAULT_BETA_2 = tune.uniform(0.1, 0.999)
    DEFAULT_EPSILON = tune.loguniform(1e-9, 1e-7)
    DEFAULT_N_ITER_NO_CHANGE = tune.randint(5, 50)
    DEFAULT_MAX_FUN = tune.randint(10000, 50000)

    def __init__(self, **kwargs) -> None:
        self.parameters = {
            "hidden_layer_sizes": kwargs.get(
                "hidden_layer_sizes", self.DEFAULT_HIDDEN_LAYER_SIZES
            ),
            "activation": kwargs.get("activation", self.DEFAULT_ACTIVATION),
            "alpha": kwargs.get("alpha", self.DEFAULT_ALPHA),
            "solver": kwargs.get("solver", self.DEFAULT_SOLVER),
            "learning_rate": kwargs.get("learning_rate", self.DEFAULT_LEARNING_RATE),
            "max_iter": kwargs.get("max_iter", self.DEFAULT_MAX_ITER),
            "tol": kwargs.get("tol", self.DEFAULT_TOL),
            "momentum": kwargs.get("momentum", self.DEFAULT_MOMENTUM),
            "nesterovs_momentum": kwargs.get(
                "nesterovs_momentum", self.DEFAULT_NESTEROVS_MOMENTUM
            ),
            "beta_1": kwargs.get("beta_1", self.DEFAULT_BETA_1),
            "beta_2": kwargs.get("beta_2", self.DEFAULT_BETA_2),
            "epsilon": kwargs.get("epsilon", self.DEFAULT_EPSILON),
            "n_iter_no_change": kwargs.get(
                "n_iter_no_change", self.DEFAULT_N_ITER_NO_CHANGE
            ),
            "max_fun": kwargs.get("max_fun", self.DEFAULT_MAX_FUN),
        }


class MLPRegressorTrainer(ModelTrainer):
    def __init__(self, config: ModelConfigurationBase | None = None):
        if config is None:
            config = ModelConfigurationBase(MLPRegressorHyperparameterSet())
        else:
            validate_config_type(config, MLPRegressorHyperparameterSet)
        super().__init__(config)
        self.model = MLPRegressor(max_iter=1000, random_state=config.random_state)

    def fit(self, X: Any, y: Any):
        if self.model is not None:
            self.model.fit(X, y)
