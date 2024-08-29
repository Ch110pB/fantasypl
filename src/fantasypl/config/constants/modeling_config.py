"""Configs for modeling with AutoML."""

TASK: str = "regression"
MODELS: list[str] = ["lgbm", "xgboost", "rf"]
METRIC: str = "rmse"
SEED: int = 43
SPLITS_CV: int = 5
TIME_TRAINING_TEAM: int = 900
TIME_TRAINING_PLAYER: int = 600
