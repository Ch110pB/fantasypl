"""Functions for creating features for player models."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF
from fantasypl.config.models.player_gameweek import PlayerGameWeek
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.modeling_helper import get_form_data
from fantasypl.utils.save_helper import save_pandas


cols_form_for_xgoals: list[str] = ["shots_on_target", "npxg", "sca", "gca"]
cols_form_for_xassists: list[str] = ["sca", "gca", "key_passes", "pass_xa"]
cols_form_for_xyc: list[str] = ["yellow_cards", "red_cards", "fouls"]
cols_form_for_xpens: list[str] = ["pens_taken", "pens_scored"]
cols_form_for_xmins: list[str] = [
    "minutes",
    "starts",
    "npxg",
    "pass_xa",
    "progressive_passes",
    "progressive_carries",
    "tackles_won",
    "blocks",
    "interceptions",
    "clearances",
]
cols_form_for_xsaves: list[str] = ["gk_saves", "gk_psxg"]


def save_player_joined_df(
    data: pd.DataFrame,
    season: Season,
    cols_form: list[str],
    target: str,
    fname: str,
) -> None:
    """

    Args:
    ----
        data: A pandas dataframe having the entire dataset.
        season: Season.
        cols_form: List of column names for lagged features on.
        target: The target(y) column.
        fname: File name to save.

    """
    if "xmins" in fname:
        data["starts"] = data["starts"].astype(int)
    grouped_form_data: pd.DataFrame = get_form_data(
        data=data,
        cols=cols_form,
        team_or_player="player",
    )
    df_final: pd.DataFrame = data.dropna(subset=[target]).merge(
        grouped_form_data,
        how="left",
        on=["player", "date"],
        validate="1:1",
    )
    for position in ["GK", "DF", "MF", "FW"]:
        df_ = (
            df_final.loc[df_final["short_position"] == position]
            .dropna(how="any")
            .reset_index(drop=True)
        )
        df_ = df_.loc[df_[target] != 0]
        save_pandas(
            df_,
            DATA_FOLDER_FBREF
            / season.folder
            / "training/players"
            / position
            / f"{fname}.csv",
        )
    logger.info("Player model features saved for {}", target)


def get_players_training_data(season: Season) -> None:
    """

    Args:
    ----
        season: Season.

    """
    with Path.open(
        DATA_FOLDER_FBREF / season.folder / "player_matchlogs.json", "r"
    ) as f:
        list_player_matchlogs: list[PlayerGameWeek] = [
            PlayerGameWeek.model_validate(el)
            for el in json.load(f).get("player_matchlogs")
        ]
    df: pd.DataFrame = pd.DataFrame([dict(el) for el in list_player_matchlogs])
    df["player"] = [player.fbref_id for player in df["player"]]
    df["team"] = [team.fbref_id for team in df["team"]]

    save_player_joined_df(
        data=df,
        season=season,
        cols_form=cols_form_for_xgoals,
        target="npxg",
        fname="player_xgoals_features",
    )
    save_player_joined_df(
        data=df,
        season=season,
        cols_form=cols_form_for_xassists,
        target="xa",
        fname="player_xassists_features",
    )
    save_player_joined_df(
        data=df,
        season=season,
        cols_form=cols_form_for_xyc,
        target="yellow_cards",
        fname="player_xyc_features",
    )
    save_player_joined_df(
        data=df,
        season=season,
        cols_form=cols_form_for_xmins,
        target="minutes",
        fname="player_xmins_features",
    )
    save_player_joined_df(
        data=df,
        season=season,
        cols_form=cols_form_for_xsaves,
        target="gk_saves",
        fname="player_xsaves_features",
    )
    save_player_joined_df(
        data=df,
        season=season,
        cols_form=cols_form_for_xpens,
        target="pens_scored",
        fname="player_xpens_features",
    )


if __name__ == "__main__":
    get_players_training_data(Seasons.SEASON_2324.value)
