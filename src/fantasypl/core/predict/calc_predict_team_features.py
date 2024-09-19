"""Functions to predict team-level for each gameweek."""

import pickle  # noqa: S403
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from fantasypl.config.constants import (
    DATA_FOLDER_FBREF,
    MODEL_FOLDER,
    TEAM_PREDICTION_SCALING_FACTORS,
)
from fantasypl.config.schemas import Season, Seasons, Team
from fantasypl.core.train.build_features_team import (
    cols_form_for_xgoals,
    cols_form_for_xpens,
    cols_form_for_xyc,
    cols_static_against_xgoals,
    cols_static_against_xpens,
    cols_static_against_xyc,
)
from fantasypl.utils import (
    get_list_teams,
    get_team_gameweek_json_to_df,
    pad_lists,
    save_pandas,
)


if TYPE_CHECKING:
    import flaml  # type: ignore[import-untyped]
    import numpy.typing as npt
    import sklearn.compose  # type: ignore[import-untyped]


last_season: Season = Seasons.SEASON_2324.value


def build_predict_features_team(season: Season, gameweek: int) -> pd.DataFrame:
    """
    Create dataframe containing all team features.

    Parameters
    ----------
    season
        The season under process.
    gameweek
        The gameweek under process.

    Returns
    -------
        A dataframe with all the features.

    """
    df_gameweek: pd.DataFrame = pd.read_csv(
        MODEL_FOLDER
        / "predictions/team"
        / f"gameweek_{gameweek}/fixtures.csv",
    )

    df_season: pd.DataFrame = get_team_gameweek_json_to_df(season)
    df_season["team"] = [
        Team.model_validate(team).fbref_id for team in df_season["team"]
    ]
    df_season["opponent"] = [
        Team.model_validate(opponent).fbref_id
        for opponent in df_season["opponent"]
    ]
    df_season = df_season.sort_values(by=["date"], ascending=True)
    df_season = df_season[
        list(
            set(cols_form_for_xgoals)
            | set(cols_static_against_xgoals)
            | set(cols_form_for_xyc)
            | set(cols_static_against_xyc)
            | set(cols_form_for_xpens)
            | set(cols_static_against_xpens)
            | {"team", "opponent"},
        )
    ]

    df_prev: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FBREF / last_season.folder / "team_seasonal_stats.csv",
    )
    df_prev["team"] = [
        next(el.fbref_id for el in get_list_teams() if el.fbref_name == x)
        for x in df_prev["team"]
    ]
    df_prev = df_prev.set_index("team")

    cols: list[str] = list(set(df_season.columns) - {"team", "opponent"})
    df_agg: pd.DataFrame = (
        df_season.groupby("team")[cols]
        .agg(lambda x: list(x)[-5:])
        .reset_index()
    )
    for col in cols:
        df_agg[col] = df_agg.apply(
            lambda row, c=col: pad_lists(row, df_prev, c, "team"),
            axis=1,
        )
    df_agg = df_agg.set_index("team")

    new_columns_: dict[str, pd.Series] = {}  # type: ignore[type-arg]
    for col in cols_form_for_xgoals + cols_form_for_xyc + cols_form_for_xpens:
        for i in range(1, 6):
            new_columns_[f"{col}_lag_{i}_for"] = df_gameweek["team"].apply(
                lambda x, idx=i, c=col: df_agg.at[x, c][1 - idx]  # noqa: PD008
            )
    df_result: pd.DataFrame = (
        pd.concat(
            [df_gameweek, pd.DataFrame(new_columns_, index=df_gameweek.index)],
            axis=1,
        )
        if new_columns_
        else df_gameweek
    )
    for col in (
        cols_static_against_xgoals
        + cols_static_against_xyc
        + cols_static_against_xpens
    ):
        df_result[f"{col}_mean_opp"] = df_result["opponent"].apply(
            lambda x, c=col: statistics.mean(df_agg.at[x, c])  # noqa: PD008
        )

    return df_result


def predict_for_stat_team(
    features: pd.DataFrame,
    target: str,
    gameweek: int,
) -> None:
    """
    Save the team stat predictions.

    Parameters
    ----------
    features
        A pandas dataframe with all the features.
    target
        The prediction target.
    gameweek
        The gameweek under process.

    """
    with Path.open(
        MODEL_FOLDER / last_season.folder / f"model_team_{target}/model.pkl",
        "rb",
    ) as fl:
        model: flaml.AutoML = pickle.load(fl)  # noqa: S301
    with Path.open(
        MODEL_FOLDER
        / last_season.folder
        / f"model_team_{target}/preprocessor.pkl",
        "rb",
    ) as fl:
        preprocessor: sklearn.compose.ColumnTransformer = pickle.load(fl)  # noqa: S301

    final_features: npt.NDArray[np.float32] = preprocessor.transform(features)
    predictions: npt.NDArray[np.float32] = model.predict(final_features)
    predictions_rescaled: npt.NDArray[np.float32] = (
        TEAM_PREDICTION_SCALING_FACTORS[target]["mean"]
        + (predictions - np.mean(predictions))
        * TEAM_PREDICTION_SCALING_FACTORS[target]["std"]
        / np.std(predictions)
    )
    features[target] = predictions_rescaled
    fpath: Path = (
        MODEL_FOLDER
        / "predictions/team"
        / f"gameweek_{gameweek}"
        / f"prediction_{target}.csv"
    )
    save_pandas(features[["team", "opponent", "gameweek", target]], fpath)
    logger.info("Predictions saved for team {}", target)


if __name__ == "__main__":
    gw: int = 5
    this_season: Season = Seasons.SEASON_2425.value
    df_features: pd.DataFrame = build_predict_features_team(this_season, gw)
    predict_for_stat_team(df_features, "xgoals", gw)
    predict_for_stat_team(df_features, "xyc", gw)
    predict_for_stat_team(df_features, "xpens", gw)
