"""Functions to predict team-level for each gameweek."""

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
from fantasypl.config.models.season import Season, Seasons
from fantasypl.config.models.team_gameweek import TeamGameweek
from fantasypl.core.train.build_features_team import (
    cols_form_for_xgoals,
    cols_form_for_xpens,
    cols_form_for_xyc,
    cols_static_against_xgoals,
    cols_static_against_xpens,
    cols_static_against_xyc,
)
from fantasypl.utils.prediction_helper import (
    list_teams,
    pad_lists,
    process_gameweek_data,
)
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

    with Path.open(
        DATA_FOLDER_FBREF / season.folder / "team_matchlogs.json", "r"
    ) as fl:
        _list_team_matchlogs: list[TeamGameweek] = [
            TeamGameweek.model_validate(el)
            for el in json.load(fl).get("team_matchlogs")
        ]
        df_season: pd.DataFrame = pd.DataFrame([
            dict(el) for el in _list_team_matchlogs
        ])
    df_season["team"] = [team.fbref_id for team in df_season["team"]]
    df_season["opponent"] = [opponent.fbref_id for opponent in df_season["opponent"]]
    df_season = df_season.sort_values(by=["date"], ascending=True)
    df_season = df_season[
        list(
            set(cols_form_for_xgoals)
            | set(cols_static_against_xgoals)
            | set(cols_form_for_xyc)
            | set(cols_static_against_xyc)
            | set(cols_form_for_xpens)
            | set(cols_static_against_xpens)
            | {"formation", "team", "opponent"}
        )
    ]

    df_prev: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FBREF / last_season.folder / "team_seasonal_stats.csv"
    )
    df_prev["team"] = [
        next(el.fbref_id for el in list_teams if el.fbref_name == x)
        for x in df_prev["team"]
    ]
    df_prev = df_prev.set_index("team")

    cols: list[str] = list(set(df_season.columns) - {"team", "opponent"})
    df_agg: pd.DataFrame = (
        df_season.groupby("team")[cols].agg(lambda x: list(x)[-5:]).reset_index()
    )
    for col in cols:
        df_agg[col] = df_agg.apply(
            lambda row, c=col: pad_lists(row, df_prev, c, "team"), axis=1
        )
    df_agg = df_agg.set_index("team")

    df_gameweek["formation"] = df_gameweek["team"].apply(
        lambda x: statistics.mode(df_agg.at[x, "formation"])  # noqa: PD008
    )
    df_gameweek["formation_vs"] = df_gameweek["opponent"].apply(
        lambda x: statistics.mode(df_agg.at[x, "formation"])  # noqa: PD008
    )
    for col in cols_form_for_xgoals + cols_form_for_xyc + cols_form_for_xpens:
        for i in range(1, 6):
            df_gameweek[f"{col}_lag_{i}_for"] = df_gameweek["team"].apply(
                lambda x, idx=i, c=col: df_agg.at[x, c][1 - idx]  # noqa: PD008
            )
    for col in (
        cols_static_against_xgoals + cols_static_against_xyc + cols_static_against_xpens
    ):
        df_gameweek[f"{col}_mean_opp"] = df_gameweek["opponent"].apply(
            lambda x, c=col: statistics.mean(df_agg.at[x, c])  # noqa: PD008
        )

    return df_gameweek


def predict_for_stat(features: pd.DataFrame, target: str, gameweek: int) -> None:
    """

    Args:
    ----
        features: The features dataframe.
        target: Prediction stat.
        gameweek: Gameweek.

    """
    with Path.open(
        MODEL_FOLDER / last_season.folder / f"model_team_{target}/model.pkl", "rb"
    ) as fl:
        model: flaml.AutoML = pickle.load(fl)
    with Path.open(
        MODEL_FOLDER / last_season.folder / f"model_team_{target}/preprocessor.pkl",
        "rb",
    ) as fl:
        preprocessor: sklearn.compose.ColumnTransformer = pickle.load(fl)

    final_features: npt.NDArray[np.float32] = preprocessor.transform(features)
    predictions: npt.NDArray[np.float32] = model.predict(final_features)
    features[target] = predictions
    fpath: Path = (
        MODEL_FOLDER
        / "predictions/team"
        / f"gameweek_{gameweek}"
        / f"prediction_{target}.csv"
    )
    save_pandas(features[["team", "opponent", "gameweek", target]], fpath)
    logger.info("Predictions saved for team {}", target)


if __name__ == "__main__":
    gw = 3
    df_features: pd.DataFrame = build_predict_features(Seasons.SEASON_2425.value, gw)
    predict_for_stat(df_features, "xgoals", gw)
    predict_for_stat(df_features, "xyc", gw)
    predict_for_stat(df_features, "xpens", gw)
