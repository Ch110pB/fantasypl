"""Functions for creating player models."""

import pickle
from typing import TYPE_CHECKING

from flaml import AutoML  # type: ignore[import-untyped]
from loguru import logger
from sklearn.metrics import (  # type: ignore[import-untyped]
    root_mean_squared_error,
)

from fantasypl.config.constants import (
    METRIC,
    MODEL_FOLDER,
    MODELS,
    SEED,
    SPLITS_CV,
    TASK,
    TIME_TRAINING_PLAYER,
)
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import get_train_test_data, save_pkl


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt


def train_model_automl(season: Season, position: str, target: str) -> None:
    """

    Parameters
    ----------
    season
        The season under process.
    position
        FBRef short position for models.
    target
        The target(y) column.

    """
    automl = AutoML()
    x_train, y_train, x_test, y_test = get_train_test_data(
        folder=f"{position}/model_player_{target}", season=season
    )
    automl.fit(
        x_train,
        y_train,
        task=TASK,
        metric=METRIC,
        estimator_list=MODELS,
        ensemble=False,
        eval_method="cv",
        n_splits=SPLITS_CV,
        split_type="uniform",
        retrain_full=True,
        seed=SEED,
        time_budget=TIME_TRAINING_PLAYER,
        early_stop=True,
        verbose=3,
        log_file_name=f"{MODEL_FOLDER}/{season.folder}/{position}/"
        f"model_player_{target}/model.log",
    )
    y_pred: npt.NDArray[np.float32] = automl.predict(x_test)
    logger.info(
        "RMSE for player {} model: {}",
        target,
        root_mean_squared_error(y_test, y_pred),
    )
    fpath: Path = (
        MODEL_FOLDER
        / season.folder
        / position
        / f"model_player_{target}/model.pkl"
    )
    save_pkl(automl, fpath, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "Model training completed for player {} for position {}",
        target,
        position,
    )


if __name__ == "__main__":
    pos_: str
    for pos_ in ["GK"]:
        train_model_automl(Seasons.SEASON_2324.value, pos_, "xsaves")
    for pos_ in ["MF", "FW"]:
        train_model_automl(Seasons.SEASON_2324.value, pos_, "xpens")
    for pos_ in ["DF", "MF", "FW"]:
        train_model_automl(Seasons.SEASON_2324.value, pos_, "xgoals")
        train_model_automl(Seasons.SEASON_2324.value, pos_, "xassists")
    for pos_ in ["GK", "DF", "MF", "FW"]:
        train_model_automl(Seasons.SEASON_2324.value, pos_, "xmins")
        train_model_automl(Seasons.SEASON_2324.value, pos_, "xyc")
