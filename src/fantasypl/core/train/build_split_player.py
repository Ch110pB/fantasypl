"""Functions for creating player train-test splits and preprocessing."""

import pandas as pd
from loguru import logger

from fantasypl.config.constants import DATA_FOLDER_FBREF
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import (
    get_team_gameweek_json_to_df,
    preprocess_data_and_save,
)


def build_split_player(
    season: Season,
    position: str,
    target_name: str,
    target_col: str,
) -> None:
    """
    Save player model train-test splits and preprocessor.

    Parameters
    ----------
    season
        The season under process.
    position
        FBRef short position for models.
    target_name
        The model name.
    target_col
        The target(y) column.

    """
    df_features: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FBREF
        / season.folder
        / "training/players"
        / position
        / f"player_{target_name}_features.csv",
    )
    if target_name == "xsaves":
        team_df: pd.DataFrame = get_team_gameweek_json_to_df(season)
        team_df["team"] = [team.fbref_id for team in team_df["team"]]
        team_df["date"] = team_df["date"].astype(str)
        team_df = team_df[["team", "date", "npxg_vs"]]
        df_features = df_features.merge(
            team_df,
            on=["team", "date"],
            how="left",
            validate="m:1",
        )
    _select_cols: list[str] = [
        col
        for col in df_features.columns
        if ("_lag_" in col) or (col == "venue")
    ]
    _add_select_cols: list[str]
    match target_name:
        case "xgoals" | "xassists" | "xyc" | "xpens" | "xmins":
            _add_select_cols = [target_col]
        case "xsaves":
            _add_select_cols = [target_col, "npxg_vs"]
        case _:
            _add_select_cols = []

    df_pd: pd.DataFrame = df_features[_select_cols + _add_select_cols]
    categorical_features: list[str] = ["venue"]
    categories: list[list[str]] = [
        df_pd[feature].unique().tolist() for feature in categorical_features
    ]
    preprocess_data_and_save(
        df=df_pd,
        target_col=target_col,
        target_name=target_name,
        categorical_features=categorical_features,
        categories=categories,
        team_or_player="player",
        season=season,
        position=position,
    )
    logger.info(
        "Train-test splits and preprocessor saved for "
        "player {} and position {}",
        target_name,
        position,
    )


if __name__ == "__main__":
    pos_: str
    for pos_ in ["GK"]:
        build_split_player(
            Seasons.SEASON_2324.value,
            pos_,
            "xsaves",
            "gk_saves",
        )
    for pos_ in ["MF", "FW"]:
        build_split_player(
            Seasons.SEASON_2324.value,
            pos_,
            "xpens",
            "pens_scored",
        )
    for pos_ in ["DF", "MF", "FW"]:
        build_split_player(
            Seasons.SEASON_2324.value,
            pos_,
            "xgoals",
            "npxg",
        )
        build_split_player(
            Seasons.SEASON_2324.value,
            pos_,
            "xassists",
            "xa",
        )
    for pos_ in ["GK", "DF", "MF", "FW"]:
        build_split_player(
            Seasons.SEASON_2324.value,
            pos_,
            "xmins",
            "minutes",
        )
        build_split_player(
            Seasons.SEASON_2324.value,
            pos_,
            "xyc",
            "yellow_cards",
        )
