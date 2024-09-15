"""Functions for creating team models."""

import pickle  # noqa: S403
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
    TIME_TRAINING_TEAM,
)
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import get_train_test_data, save_pkl


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import numpy.typing as npt


def train_model_automl(season: Season, target: str) -> None:
    """
    Train team models.

    Parameters
    ----------
    season
        The season under process.
    target
        The target(y) column.

    """
    automl = AutoML()
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
        ensemble=False,
        eval_method="cv",
        n_splits=SPLITS_CV,
        split_type="uniform",
        retrain_full=True,
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
    fpath: Path = (
        MODEL_FOLDER / season.folder / f"model_team_{target}/model.pkl"
    )
    save_pkl(automl, fpath, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Model training completed for team {}", target)


if __name__ == "__main__":
    train_model_automl(Seasons.SEASON_2324.value, "xgoals")
    train_model_automl(Seasons.SEASON_2324.value, "xyc")
    train_model_automl(Seasons.SEASON_2324.value, "xpens")
