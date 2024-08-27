"""Functions for creating teams and players dataframes from FPL API data."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FPL
from fantasypl.config.models.season import Season, Seasons
from fantasypl.config.references.player_refs import FBREF_FPL_PLAYER_REF_DICT
from fantasypl.utils.save_helper import save_pandas


_cols_teams: list[str] = ["id", "code", "name", "short_name"]
_cols_players: list[str] = [
    "id",
    "code",
    "full_name",
    "web_name",
    "photo",
    "team",
    "team_code",
    "element_type",
    "now_cost",
    "chance_of_playing_next_round",
    "chance_of_playing_this_round",
    "news",
    "news_added",
    "selected_by_percent",
]


def save_teams(season: Season) -> None:
    """

    Args:
    ----
        season: Season.

    """
    with Path.open(DATA_FOLDER_FPL / season.folder / "bootstrap.json", "r") as f:
        list_teams_dicts: list[dict[str, Any]] | None = json.load(f).get("teams")
    if not list_teams_dicts:
        logger.error("The key `team` not present in FPL bootstrap")
        return
    df_teams: pd.DataFrame = pd.DataFrame(list_teams_dicts)
    df_teams = df_teams[_cols_teams]
    fpath: Path = DATA_FOLDER_FPL / season.folder / "teams.csv"
    save_pandas(df=df_teams, fpath=fpath)


def save_players(season: Season) -> None:
    """

    Args:
    ----
        season: Season.

    """
    with Path.open(DATA_FOLDER_FPL / season.folder / "bootstrap.json", "r") as f:
        list_players_dicts: list[dict[str, Any]] | None = json.load(f).get("elements")
    if not list_players_dicts:
        logger.error("The key `elements` not present in FPL bootstrap")
        return
    df_players: pd.DataFrame = pd.DataFrame(list_players_dicts)
    df_players["full_name"] = df_players["first_name"] + " " + df_players["second_name"]
    df_players = df_players[_cols_players]
    fpath: Path = DATA_FOLDER_FPL / season.folder / "players.csv"
    logger.info(
        "Players missing in refs from season {}: {}",
        season.fbref_name,
        [
            *set(df_players["code"].tolist())
            - set(*[FBREF_FPL_PLAYER_REF_DICT.values()])
        ],
    )
    save_pandas(df=df_players, fpath=fpath)


if __name__ == "__main__":
    # save_teams(Season.SEASON_2324)
    save_teams(Seasons.SEASON_2425.value)
    # save_players(Season.SEASON_2324)
    save_players(Seasons.SEASON_2425.value)
