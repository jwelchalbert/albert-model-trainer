from typing import Dict, Any, List, Tuple
from albert_model_trainer.base.metrics.performance_metric import PerformanceMetrics
from albert_model_trainer.base.hyperparameter import HyperParameterSet
from sklearn.base import BaseEstimator


# 1. Abstract ModelTrainer class
class ModelTrainer:
    def __init__(self, hyperparameters: HyperParameterSet):
        self.hyperparameters = hyperparameters
        self.model: BaseEstimator | None = None
        self.optimal_parameters = {}

    def fit(self, X: Any, y: Any | None = None):
        raise NotImplementedError

    def evaluate(self) -> PerformanceMetrics:
        raise NotImplementedError

    def set_hyperparameters(self, hyperparameters: HyperParameterSet):
        self.hyperparameters = hyperparameters


    def tune(self):
        if self.model is None:
            raise ValueError("no model instance is set -- nothing to tune")
        
        