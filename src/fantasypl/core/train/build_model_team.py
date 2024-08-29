"""Functions for creating team models."""

import pickle
from typing import TYPE_CHECKING

from flaml import AutoML  # type: ignore[import-untyped]
from loguru import logger
from sklearn.metrics import root_mean_squared_error

from fantasypl.config.constants.folder_config import MODEL_FOLDER
from fantasypl.config.constants.modeling_config import (
    METRIC,
    MODELS,
    SEED,
    SPLITS_CV,
    TASK,
    TIME_TRAINING_TEAM,
)
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.modeling_helper import get_train_test_data
from fantasypl.utils.save_helper import save_pkl


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt


def train_model_automl(season: Season, target: str) -> None:
    """

    Args:
    ----
        season: Season.
        target: The target(y) column.

    """
    automl = AutoML()
    x_train: npt.NDArray[np.float32]
    y_train: npt.NDArray[np.float32]
    x_test: npt.NDArray[np.float32]
    y_test: npt.NDArray[np.float32]
    x_train, y_train, x_test, y_test = get_train_test_data(
        folder=f"model_team_{target}",
        season=season,
    )
    automl.fit(
        x_train,
        y_train,
        task=TASK,
        metric=METRIC,
        estimator_list=MODELS,
        ensemble=True,
        eval_method="cv",
        n_splits=SPLITS_CV,
        split_type="uniform",
        seed=SEED,
        time_budget=TIME_TRAINING_TEAM,
        early_stop=True,
        verbose=3,
        log_file_name=f"{MODEL_FOLDER}/{season.folder}/model_team_{target}/model.log",
    )
    y_pred: npt.NDArray[np.float32] = automl.predict(x_test)
    logger.info(
        "RMSE for team {} model: {}",
        target,
        root_mean_squared_error(y_test, y_pred),
    )
    fpath: Path = MODEL_FOLDER / season.folder / f"model_team_{target}/model.pkl"
    save_pkl(automl, fpath, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model training completed for team {}", target)


if __name__ == "__main__":
    train_model_automl(Seasons.SEASON_2324.value, "xgoals")
    train_model_automl(Seasons.SEASON_2324.value, "xyc")
    train_model_automl(Seasons.SEASON_2324.value, "xpens")
