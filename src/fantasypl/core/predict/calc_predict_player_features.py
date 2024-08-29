"""Functions to predict player-level for each gameweek."""

import json
import pickle
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from fantasypl.config.constants.folder_config import (
    DATA_FOLDER_FBREF,
    MODEL_FOLDER,
)
from fantasypl.config.models.player_gameweek import PlayerGameWeek
from fantasypl.config.models.season import Season, Seasons
from fantasypl.core.train.build_features_player import (
    cols_form_for_xassists,
    cols_form_for_xgoals,
    cols_form_for_xmins,
    cols_form_for_xpens,
    cols_form_for_xsaves,
    cols_form_for_xyc,
)
from fantasypl.utils.prediction_helper import pad_lists, process_gameweek_data
from fantasypl.utils.save_helper import save_pandas


if TYPE_CHECKING:
    import flaml  # type: ignore[import-untyped]
    import numpy as np
    import numpy.typing as npt
    import sklearn.compose


last_season: Season = Seasons.SEASON_2324.value


def build_predict_features(season: Season, gameweek: int) -> pd.DataFrame:
    """

    Args:
    ----
        season: Season.
        gameweek: Gameweek.

    Returns:
    -------
        A dataframe with all the features.

    """
    df_gameweek = process_gameweek_data(gameweek)

    df_xgoals: pd.DataFrame = pd.read_csv(
        MODEL_FOLDER / "predictions/team" / f"gameweek_{gameweek}/prediction_xgoals.csv"
    )
    dict_xgoals: dict[tuple[str, str, int], float] = df_xgoals.set_index([
        "team",
        "opponent",
        "gameweek",
    ]).to_dict()["xgoals"]
    df_xgoals["npxg_vs"] = df_xgoals.apply(
        lambda row: dict_xgoals[row["opponent"], row["team"], row["gameweek"]], axis=1
    )
    df_gameweek = df_gameweek.merge(
        df_xgoals, on=["team", "opponent", "gameweek"], how="left"
    )

    with Path.open(
        DATA_FOLDER_FBREF / season.folder / "player_matchlogs.json", "r"
    ) as fl:
        _list_player_matchlogs: list[PlayerGameWeek] = [
            PlayerGameWeek.model_validate(el)
            for el in json.load(fl).get("player_matchlogs")
        ]
        df_season: pd.DataFrame = pd.DataFrame([
            dict(el) for el in _list_player_matchlogs
        ])
    df_season["player"] = [player.fbref_id for player in df_season["player"]]
    df_season["team"] = [team.fbref_id for team in df_season["team"]]

    df_gameweek = df_gameweek.merge(
        df_season[["player", "team", "short_position"]],
        on="team",
        how="left",
        validate="m:m",
    )
    df_season = df_season.sort_values(by=["date"], ascending=True)
    df_season = df_season[
        list(
            set(cols_form_for_xgoals)
            | set(cols_form_for_xassists)
            | set(cols_form_for_xyc)
            | set(cols_form_for_xmins)
            | set(cols_form_for_xpens)
            | set(cols_form_for_xsaves)
            | {"player", "team", "short_position"}
        )
    ]

    df_prev: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FBREF / last_season.folder / "player_seasonal_stats.csv"
    )
    df_prev = df_prev.set_index("player")

    cols: list[str] = list(set(df_season.columns) - {"player", "team"})
    df_agg: pd.DataFrame = (
        df_season.groupby("player")[cols].agg(lambda x: list(x)[-5:]).reset_index()
    )
    for col in cols:
        df_agg[col] = df_agg.apply(
            lambda row, c=col: pad_lists(row, df_prev, c, "player"), axis=1
        )
    df_agg = df_agg.set_index("player")
    df_gameweek["short_position"] = df_gameweek["player"].apply(
        lambda x: statistics.mode(df_agg.at[x, "short_position"])  # noqa: PD008
    )
    new_columns_ = {}
    for col in (
        cols_form_for_xgoals
        + cols_form_for_xassists
        + cols_form_for_xyc
        + cols_form_for_xmins
        + cols_form_for_xpens
        + cols_form_for_xsaves
    ):
        for i in range(1, 6):
            new_columns_[f"{col}_lag_{i}"] = df_gameweek["player"].apply(
                lambda x, idx=i, c=col: df_agg.at[x, c][1 - idx]  # noqa: PD008
            )
    return (
        pd.concat(
            [df_gameweek, pd.DataFrame(new_columns_, index=df_gameweek.index)], axis=1
        )
        if new_columns_
        else df_gameweek
    )


def predict_for_stat(
    features: pd.DataFrame, position: str, target: str, gameweek: int
) -> None:
    """

    Args:
    ----
        features: The features dataframe.
        position: Player position.
        target: Prediction stat.
        gameweek: Gameweek.

    """
    with Path.open(
        MODEL_FOLDER
        / last_season.folder
        / position
        / f"model_player_{target}/model.pkl",
        "rb",
    ) as fl:
        model: flaml.AutoML = pickle.load(fl)
    with Path.open(
        MODEL_FOLDER
        / last_season.folder
        / position
        / f"model_player_{target}/preprocessor.pkl",
        "rb",
    ) as fl:
        preprocessor: sklearn.compose.ColumnTransformer = pickle.load(fl)

    features = features.loc[features["short_position"] == position]
    final_features: npt.NDArray[np.float32] = preprocessor.transform(features)
    predictions: npt.NDArray[np.float32] = model.predict(final_features)
    features.loc[:, [target]] = predictions
    fpath: Path = (
        MODEL_FOLDER
        / "predictions/player"
        / position
        / f"gameweek_{gameweek}"
        / f"prediction_{target}"
    )
    save_pandas(
        features[["player", "team", "gameweek", "short_position", target]], fpath
    )
    logger.info("Predictions saved for player {} for position {}", target, position)


if __name__ == "__main__":
    gw = 3
    df_features: pd.DataFrame = build_predict_features(Seasons.SEASON_2425.value, gw)
    for pos_ in ["GK"]:
        predict_for_stat(df_features, pos_, "xsaves", gw)
    for pos_ in ["MF", "FW"]:
        predict_for_stat(df_features, pos_, "xpens", gw)
    for pos_ in ["DF", "MF", "FW"]:
        predict_for_stat(df_features, pos_, "xgoals", gw)
        predict_for_stat(df_features, pos_, "xassists", gw)
    for pos_ in ["GK", "DF", "MF", "FW"]:
        predict_for_stat(df_features, pos_, "xmins", gw)
        predict_for_stat(df_features, pos_, "xyc", gw)
