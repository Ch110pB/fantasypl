"""Functions for creating player matchlogs for entire season."""

import json
import os
from functools import reduce
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import rich.progress
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF, DATA_FOLDER_REF
from fantasypl.config.constants.mapping_config import FBREF_POSITION_DICT
from fantasypl.config.models.player import Player
from fantasypl.config.models.player_gameweek import PlayerGameWeek
from fantasypl.config.models.season import Season, Seasons
from fantasypl.config.models.team import Team
from fantasypl.utils.modeling_helper import get_teams
from fantasypl.utils.save_helper import save_json


with Path.open(DATA_FOLDER_REF / "teams.json", "r") as f:
    _list_teams: list[Team] = [
        Team.model_validate(el) for el in json.load(f).get("teams")
    ]
with Path.open(DATA_FOLDER_FBREF / "players.json", "r") as f:
    _list_players: list[Player] = [
        Player.model_validate(el) for el in json.load(f).get("players")
    ]
_player_lookup_dict: dict[str, Player] = {el.fbref_name: el for el in _list_players}


def process_single_team(team: Team, season: Season) -> list[dict[str, PlayerGameWeek]]:
    """

    Args:
    ----
        team: Team class object.
        season: Season.

    Returns:
    -------
        A list containing all players' gameweek data for the team.

    """
    list_files: list[str] = next(
        iter(os.walk(DATA_FOLDER_FBREF / season.folder / "matches" / team.short_name))
    )[2]
    dfs_summary: list[pd.DataFrame] = []
    dfs_passing: list[pd.DataFrame] = []
    dfs_defense: list[pd.DataFrame] = []
    dfs_misc: list[pd.DataFrame] = []
    dfs_keeper: list[pd.DataFrame] = []

    for fl in list_files:
        df_stats: pd.DataFrame = pd.read_csv(
            DATA_FOLDER_FBREF / season.folder / "matches" / team.short_name / fl
        )
        df_stats["starts"] = np.where(df_stats["player"].str.contains("\xa0"), 0, 1)
        df_stats["player"] = df_stats["player"].str.strip()
        _join_cols: list[str] = ["player", "date", "venue"]
        match fl:
            case fl if "summary" in fl:
                df_stats["short_position"] = (
                    df_stats["position"].str.split(",").str[0].map(FBREF_POSITION_DICT)
                )
                df_stats = df_stats.rename(
                    columns={
                        "header_performance_shots_on_target": "shots_on_target",
                        "header_performance_cards_yellow": "yellow_cards",
                        "header_performance_cards_red": "red_cards",
                        "header_performance_pens_att": "pens_taken",
                        "header_performance_pens_made": "pens_scored",
                        "header_expected_npxg": "npxg",
                        "header_expected_xg_assist": "xa",
                        "header_sca_sca": "sca",
                        "header_sca_gca": "gca",
                        "header_carries_progressive_carries": "progressive_carries",
                    },
                )
                df_stats = df_stats[
                    [
                        *_join_cols,
                        "short_position",
                        "minutes",
                        "starts",
                        "shots_on_target",
                        "npxg",
                        "xa",
                        "sca",
                        "gca",
                        "progressive_carries",
                        "yellow_cards",
                        "red_cards",
                        "pens_taken",
                        "pens_scored",
                    ]
                ]
                dfs_summary.append(df_stats)
            case fl if "passing" in fl:
                df_stats = df_stats.rename(columns={"assisted_shots": "key_passes"})
                df_stats = df_stats[
                    [
                        *_join_cols,
                        "key_passes",
                        "pass_xa",
                        "progressive_passes",
                    ]
                ]
                dfs_passing.append(df_stats)
            case fl if "defense" in fl:
                df_stats = df_stats.rename(
                    columns={
                        "header_tackles_tackles_won": "tackles_won",
                        "header_blocks_blocks": "blocks",
                    },
                )
                df_stats = df_stats[
                    [
                        *_join_cols,
                        "tackles_won",
                        "blocks",
                        "interceptions",
                        "clearances",
                    ]
                ]
                dfs_defense.append(df_stats)
            case fl if "misc" in fl:
                df_stats = df_stats.rename(
                    columns={"header_performance_fouls": "fouls"}
                )
                df_stats = df_stats[[*_join_cols, "fouls"]]
                dfs_misc.append(df_stats)
            case fl if "keeper" in fl:
                df_stats = df_stats.rename(
                    columns={
                        "header_gk_shot_stopping_gk_saves": "gk_saves",
                        "header_gk_shot_stopping_gk_psxg": "gk_psxg",
                    },
                )
                df_stats = df_stats[[*_join_cols, "gk_saves", "gk_psxg"]]
                dfs_keeper.append(df_stats)
            case _:
                logger.error("Untracked file: {}", fl)

    df_summary: pd.DataFrame = (
        pd.concat(dfs_summary, ignore_index=True) if dfs_summary else pd.DataFrame()
    )
    df_passing: pd.DataFrame = (
        pd.concat(dfs_passing, ignore_index=True) if dfs_passing else pd.DataFrame()
    )
    df_defense: pd.DataFrame = (
        pd.concat(dfs_defense, ignore_index=True) if dfs_defense else pd.DataFrame()
    )
    df_misc: pd.DataFrame = (
        pd.concat(dfs_misc, ignore_index=True) if dfs_misc else pd.DataFrame()
    )
    df_keeper: pd.DataFrame = (
        pd.concat(dfs_keeper, ignore_index=True) if dfs_keeper else pd.DataFrame()
    )

    df_final: pd.DataFrame = reduce(
        lambda left, right: left.merge(
            right,
            on=_join_cols,
            how="left",
            validate="m:m",
        ),
        [df_summary, df_passing, df_defense, df_misc, df_keeper],
    )
    df_final = df_final.fillna(0)
    df_final["player"] = df_final["player"].map(_player_lookup_dict)
    return [
        PlayerGameWeek.model_validate(
            {"team": team, "season": season.fbref_long_name, **row},
        ).model_dump()
        for row in df_final.to_dict(orient="records")
    ]


def save_aggregate_player_matchlogs(
    season: Literal[Seasons.SEASON_2324, Seasons.SEASON_2425],
) -> None:
    """

    Args:
    ----
        season: Season.

    """
    dfs: list[dict[str, PlayerGameWeek]] = []
    _teams: list[str] = get_teams(season.value)
    for team_name in rich.progress.track(_teams):
        team: Team = next(el for el in _list_teams if el.fbref_name == team_name)
        df_temp: list[dict[str, PlayerGameWeek]] = process_single_team(
            team,
            season.value,
        )
        dfs += df_temp
    fpath: Path = DATA_FOLDER_FBREF / season.value.folder / "player_matchlogs.json"
    save_json({"player_matchlogs": dfs}, fpath=fpath, default=str)
    logger.info(
        "Player matchlogs saved for all clubs from Season: {}",
        season.value.fbref_name,
    )


if __name__ == "__main__":
    # save_aggregate_player_matchlogs(Seasons.SEASON_2324)
    save_aggregate_player_matchlogs(Seasons.SEASON_2425)
