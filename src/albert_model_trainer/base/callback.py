# trunk-ignore-all(ruff/F821)

from typing import Dict

from albert_model_trainer.base.metrics import (
    AggregatePerformanceMetrics, NamedAggregatePerformanceMetrics,
    PerformanceMetrics)


class Callback:
    @property
    def state_key(self) -> str:
        return self.__class__.__qualname__

    def setup(
        self,
        trainer: "ModelTrainer",
    ) -> None:
        """Signals when tune begins."""

    def teardown(
        self,
        trainer: "ModelTrainer",
    ) -> None:
        """Signals when tune ends."""

    def on_ray_pre_init(self, trainer: "ModelTrainer") -> None:
        """Signals right before the ray-init call -- can be used to set the state of the trainer ray args if desired."""

    def on_ray_pre_shutdown(self, trainer: "ModelTrainer") -> None:
        """Signals right before ray.shutdown() not sure what this will be useful for."""

    def on_tune_start(
        self,
        trainer: "ModelTrainer",
        model_idx: int,
        num_features: int,
    ) -> None:
        """Signals when tune starts on a given Model Training Operation."""

    def on_tune_validation_end(
        self,
        trainer: "ModelTrainer",
        metrics: PerformanceMetrics
        | AggregatePerformanceMetrics
        | NamedAggregatePerformanceMetrics,
    ):
        """Signals when a tune step has evaluated the validation metrics."""

    def on_tune_train_end(
        self,
        trainer: "ModelTrainer",
        metrics: PerformanceMetrics
        | AggregatePerformanceMetrics
        | NamedAggregatePerformanceMetrics,
    ) -> None:
        """Signals when a tune step has evaluated the training metrics."""

    def on_tune_step_begin(self, trainer: "ModelTrainer") -> None:
        """Signals when a single hyperparameter tune step begins."""

    def on_tune_step_complete(self, trainer: "ModelTrainer") -> None:
        """Signals when a single hyperparameter tune step has completed during a tune session."""

    def on_tune_multi_output_end(
        self,
        trainer: "ModelTrainer",
        output_num: int,
    ) -> None:
        """Signals when a multi output hyperparameter tune session has completed."""

    def on_tune_end(self, trainer: "ModelTrainer", model_idx: int) -> None:
        """Signals when tune stops/completes on a given model."""

    def on_model_build_start(self, trainer: "ModelTrainer", model_idx: int) -> None:
        """Signals when a model build starts."""

    def on_model_build_end(self, trainer: "ModelTrainer", model_idx: int) -> None:
        """Signals when a model build ends."""


class CallbackInvoker:
    def __init__(self, callbacks: list[Callback]) -> None:
        self.callbacks = callbacks

    def trigger_callback(self, method: str, args: Dict = None):
        if args is None:
            args = {}

        def call_method_on_callback(callback):
            getattr(callback, method)(**args)

        if self.callbacks is not None:
            list(map(call_method_on_callback, self.callbacks))
