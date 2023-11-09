import pandas as pd
from sklearn.datasets import make_regression

from albert_model_trainer.auto_fit.fitter import SklearnAutoRegressor
from albert_model_trainer.models.model_iterator import get_all_model_trainers

# Generate some simple random regression data
X, y = make_regression(n_samples=200, n_features=20, noise=0.1, random_state=42)

# Fit a set of models to the data
builder = SklearnAutoRegressor(
    "r2",
    ["mae", "mse", "r2"],
    5,
    get_all_model_trainers()[:2],
    random_state=42,
    scaling_columns=[i for i in range(20)],
    num_hyperopt_samples=5,
)
builder.fit(X, y)

pipeline = builder.result_tracker.get_nth_top_model_pipeline()
print(pipeline)

X, y2 = make_regression(n_samples=200, n_features=20, n_targets=2, noise=0.1)

builder = SklearnAutoRegressor(
    "r2",
    ["mae", "mse", "r2"],
    5,
    get_all_model_trainers()[:2],
    random_state=42,
    scaling_columns=[i for i in range(20)],
    num_hyperopt_samples=5,
)

builder.fit(X, y2)
pipeline = builder.result_tracker.get_nth_top_model_pipeline()
print(pipeline)
