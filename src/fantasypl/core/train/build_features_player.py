"""Functions for creating features for the player models."""

import statistics

import pandas as pd
from loguru import logger

from fantasypl.config.constants import DATA_FOLDER_FBREF
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import (
    get_form_data,
    get_player_gameweek_json_to_df,
    save_pandas,
)


cols_form_for_xgoals: list[str] = ["shots_on_target", "npxg", "sca", "gca"]
cols_form_for_xassists: list[str] = ["sca", "gca", "key_passes", "pass_xa"]
cols_form_for_xyc: list[str] = ["yellow_cards", "red_cards", "fouls"]
cols_form_for_xpens: list[str] = ["pens_taken", "pens_scored"]
cols_form_for_xmins: list[str] = [
    "minutes",
    "starts",
    "npxg",
    "pass_xa",
    "progressive_actions",
    "defensive_actions",
]
cols_form_for_xsaves: list[str] = ["gk_saves", "gk_psxg"]


def save_player_joined_df(
    data: pd.DataFrame, season: Season, cols_form: list[str], stat: str
) -> None:
    """

    Parameters
    ----------
    data
        A pandas dataframe containing full stats.
    season
        The season under process.
    cols_form
        The columns to create lagged features.
    stat
        The model name.

    """
    if stat == "xmins":
        data["starts"] = data["starts"].astype(int)
        data["progressive_actions"] = (
            data["progressive_carries"] + data["progressive_passes"]
        )
        data["defensive_actions"] = (
            data["tackles_won"]
            + data["blocks"]
            + data["interceptions"]
            + data["clearances"]
        )
    grouped_form_data: pd.DataFrame = get_form_data(
        data=data, cols=cols_form, team_or_player="player"
    )
    df_final: pd.DataFrame = data.merge(
        grouped_form_data, how="left", on=["player", "date"], validate="m:m"
    )
    positions: dict[str, str] = (
        df_final.groupby("player")["short_position"]
        .agg(list)
        .apply(lambda x: statistics.mode([el for el in x if el is not None]))
        .to_dict()
    )
    df_final["short_position"] = df_final["short_position"].fillna(
        df_final["player"].map(positions)
    )
    for position in ["GK", "DF", "MF", "FW"]:
        df_ = (
            df_final.loc[df_final["short_position"] == position]
            .dropna(how="any")
            .reset_index(drop=True)
        )
        save_pandas(
            df_,
            DATA_FOLDER_FBREF
            / season.folder
            / "training/players"
            / position
            / f"player_{stat}_features.csv",
        )
    logger.info("Player model features saved for {}", stat)


def get_players_training_data(season: Season) -> None:
    """

    Parameters
    ----------
    season
        The season under process.

    """
    player_df: pd.DataFrame = get_player_gameweek_json_to_df(season)
    player_df["player"] = [player.fbref_id for player in player_df["player"]]
    player_df["team"] = [team.fbref_id for team in player_df["team"]]

    save_player_joined_df(
        data=player_df,
        season=season,
        cols_form=cols_form_for_xgoals,
        stat="xgoals",
    )
    save_player_joined_df(
        data=player_df,
        season=season,
        cols_form=cols_form_for_xassists,
        stat="xassists",
    )
    save_player_joined_df(
        data=player_df,
        season=season,
        cols_form=cols_form_for_xyc,
        stat="xyc",
    )
    save_player_joined_df(
        data=player_df,
        season=season,
        cols_form=cols_form_for_xmins,
        stat="xmins",
    )
    save_player_joined_df(
        data=player_df,
        season=season,
        cols_form=cols_form_for_xsaves,
        stat="xsaves",
    )
    save_player_joined_df(
        data=player_df,
        season=season,
        cols_form=cols_form_for_xpens,
        stat="xpens",
    )


if __name__ == "__main__":
    get_players_training_data(Seasons.SEASON_2324.value)
