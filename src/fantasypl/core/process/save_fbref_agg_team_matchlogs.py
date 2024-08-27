"""Functions for creating team matchlogs for entire season."""

import json
from functools import reduce
from pathlib import Path
from typing import Literal

import pandas as pd
import rich.progress
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF, DATA_FOLDER_REF
from fantasypl.config.models.season import Season, Seasons
from fantasypl.config.models.team import Team
from fantasypl.config.models.team_gameweek import TeamGameweek
from fantasypl.utils.modeling_helper import get_teams
from fantasypl.utils.save_helper import save_json


with Path.open(DATA_FOLDER_REF / "teams.json", "r") as f:
    _list_teams: list[Team] = [
        Team.model_validate(el) for el in json.load(f).get("teams")
    ]


def process_single_team(
    team_short_name: str,
    season: Season,
    last_season_flag: bool = False,
) -> list[dict[str, TeamGameweek]]:
    """

    Args:
    ----
        team_short_name: Team FPL API short name.
        season: Season.
        last_season_flag: True if data required for last season, False otherwise.

    Returns:
    -------
        A list containing teams' gameweek data for the team.

    """
    folder_structure: str = (
        f"{DATA_FOLDER_FBREF}/{season.folder}/"
        f"team_matchlogs/{team_short_name}/{{}}.csv"
    )
    if last_season_flag:
        folder_structure = (
            f"{DATA_FOLDER_FBREF}/{Seasons.SEASON_2324.value.folder}/"
            f"team_matchlogs/{team_short_name}/{{}}.csv"
        )

    df_schedule_for: pd.DataFrame = pd.read_csv(folder_structure.format("schedule_for"))
    df_schedule_for = df_schedule_for[
        ["date", "venue", "result", "possession", "formation", "opp_formation"]
    ].rename(columns={"opp_formation": "formation_vs"})
    df_schedule_for = df_schedule_for.dropna(subset=["date", "result"], how="any").drop(
        columns="result",
    )
    df_schedule_for = df_schedule_for.loc[~df_schedule_for[["date"]].eq("").any(axis=1)]

    df_shooting_for: pd.DataFrame = pd.read_csv(
        folder_structure.format("shooting_for"),
    ).rename(
        columns={
            "header_for_against_date": "date",
            "header_standard_shots_on_target": "shots_on_target",
            "header_standard_pens_att": "pens_won",
            "header_standard_pens_made": "pens_scored",
            "header_expected_npxg": "npxg",
        },
    )
    df_shooting_for = df_shooting_for[
        ["date", "shots_on_target", "pens_won", "pens_scored", "npxg"]
    ].dropna(subset="date")
    df_shooting_for = df_shooting_for.loc[~df_shooting_for[["date"]].eq("").any(axis=1)]

    df_shooting_vs: pd.DataFrame = pd.read_csv(
        folder_structure.format("shooting_against"),
    ).rename(
        columns={
            "header_for_against_date": "date",
            "header_expected_npxg": "npxg_vs",
        },
    )
    df_shooting_vs = df_shooting_vs[["date", "npxg_vs"]].dropna(subset="date")
    df_shooting_vs = df_shooting_vs.loc[~df_shooting_vs[["date"]].eq("").any(axis=1)]

    df_passing_for: pd.DataFrame = pd.read_csv(
        folder_structure.format("passing_for"),
    ).rename(
        columns={
            "header_for_against_date": "date",
            "assisted_shots": "key_passes",
        },
    )
    df_passing_for = df_passing_for[["date", "key_passes", "pass_xa"]].dropna(
        subset="date",
    )
    df_passing_for = df_passing_for.loc[~df_passing_for[["date"]].eq("").any(axis=1)]

    df_defense_vs: pd.DataFrame = pd.read_csv(
        folder_structure.format("defense_against"),
    ).rename(
        columns={
            "header_for_against_date": "date",
            "header_tackles_tackles_won": "tackles_won_vs",
            "header_blocks_blocks": "blocks_vs",
            "interceptions": "interceptions_vs",
            "clearances": "clearances_vs",
        },
    )
    df_defense_vs = df_defense_vs[
        [
            "date",
            "tackles_won_vs",
            "blocks_vs",
            "interceptions_vs",
            "clearances_vs",
        ]
    ].dropna(subset="date")
    df_defense_vs = df_defense_vs.loc[~df_defense_vs[["date"]].eq("").any(axis=1)]

    df_keeper_vs: pd.DataFrame = pd.read_csv(
        folder_structure.format("keeper_against"),
    ).rename(
        columns={
            "header_for_against_date": "date",
            "header_performance_gk_saves": "gk_saves_vs",
        },
    )
    df_keeper_vs = df_keeper_vs[["date", "gk_saves_vs"]].dropna(subset="date")
    df_keeper_vs = df_keeper_vs.loc[~df_keeper_vs[["date"]].eq("").any(axis=1)]

    df_gca_for: pd.DataFrame = pd.read_csv(folder_structure.format("gca_for")).rename(
        columns={
            "header_for_against_date": "date",
            "header_sca_types_sca": "sca",
            "header_gca_types_gca": "gca",
        },
    )
    df_gca_for = df_gca_for[["date", "sca", "gca"]].dropna(subset="date")
    df_gca_for = df_gca_for.loc[~df_gca_for[["date"]].eq("").any(axis=1)]

    df_misc_for: pd.DataFrame = pd.read_csv(folder_structure.format("misc_for")).rename(
        columns={
            "header_for_against_date": "date",
            "header_performance_cards_yellow": "yellow_cards",
            "header_performance_cards_red": "red_cards",
            "header_performance_fouls": "fouls_conceded",
        },
    )
    df_misc_for = df_misc_for[
        ["date", "yellow_cards", "red_cards", "fouls_conceded"]
    ].dropna(subset="date")
    df_misc_for = df_misc_for.loc[~df_misc_for[["date"]].eq("").any(axis=1)]

    df_misc_vs: pd.DataFrame = pd.read_csv(
        folder_structure.format("misc_against"),
    ).rename(
        columns={
            "header_for_against_date": "date",
            "header_performance_cards_yellow": "yellow_cards_opp_vs",
            "header_performance_cards_red": "red_cards_opp_vs",
            "header_performance_fouled": "fouls_won_vs",
            "header_performance_pens_conceded": "pens_conceded_vs",
        },
    )
    df_misc_vs = df_misc_vs[
        [
            "date",
            "yellow_cards_opp_vs",
            "red_cards_opp_vs",
            "fouls_won_vs",
            "pens_conceded_vs",
        ]
    ].dropna(subset="date")
    df_misc_vs = df_misc_vs.loc[~df_misc_vs[["date"]].eq("").any(axis=1)]

    df_teamgw: pd.DataFrame = reduce(
        lambda left, right: left.merge(
            right,
            on="date",
            how="left",
            validate="1:1",
        ),
        [
            df_schedule_for,
            df_shooting_for,
            df_shooting_vs,
            df_passing_for,
            df_defense_vs,
            df_keeper_vs,
            df_gca_for,
            df_misc_for,
            df_misc_vs,
        ],
    )

    team: Team = next(el for el in _list_teams if el.short_name == team_short_name)
    df_teamgw = df_teamgw.sort_values(by="date", ascending=True)
    return [
        TeamGameweek.model_validate(
            {"team": team, "season": season.fbref_long_name, **row},
        ).model_dump()
        for row in df_teamgw.to_dict(orient="records")
    ]


def save_aggregate_team_matchlogs(
    season: Literal[Seasons.SEASON_2324, Seasons.SEASON_2425],
    last_season_flag: bool = False,
) -> None:
    """

    Args:
    ----
        season: Season.
        last_season_flag: True if data required for last season, False otherwise.

    """
    dfs: list[dict[str, TeamGameweek]] = []
    _teams: list[str] = get_teams(season.value)
    for team_name in rich.progress.track(_teams):
        team: Team = next(el for el in _list_teams if el.fbref_name == team_name)
        df_temp: list[dict[str, TeamGameweek]] = process_single_team(
            team.short_name,
            season.value,
            last_season_flag,
        )
        dfs += df_temp
    fpath_str: str = (
        "team_matchlogs_last_season.json" if last_season_flag else "team_matchlogs.json"
    )
    fpath: Path = DATA_FOLDER_FBREF / season.value.folder / fpath_str
    save_json({"team_matchlogs": dfs}, fpath=fpath, default=str)
    logger.info(
        "Team matchlogs saved for all clubs from Season: {} / Last season: {}",
        season.value.fbref_name,
        last_season_flag,
    )


if __name__ == "__main__":
    save_aggregate_team_matchlogs(Seasons.SEASON_2324)
    save_aggregate_team_matchlogs(Seasons.SEASON_2425, True)
    save_aggregate_team_matchlogs(Seasons.SEASON_2425, False)
