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
from fantasypl.utils.modeling_helper import get_fbref_teams
from fantasypl.utils.save_helper import save_json


with Path.open(DATA_FOLDER_REF / "teams.json", "r") as f:
    _list_teams: list[Team] = [
        Team.model_validate(el) for el in json.load(f).get("teams")
    ]


def process_single_stat(
    folder_structure: Path,
    stat: str,
    rename_dict: dict[str, str],
    cols: list[str],
    dropna_cols: list[str],
) -> pd.DataFrame:
    """

    Args:
    ----
        folder_structure: Path for the parent folder.
        stat: File name to look at.
        rename_dict: Column rename dictionary.
        cols: Columns to be selected.
        dropna_cols: Columns to mark empty rows.

    Returns:
    -------
        A pandas dataframe containing data for a particular stat.

    """
    df_: pd.DataFrame = pd.read_csv(folder_structure / f"{stat}.csv")
    df_ = df_.rename(columns=rename_dict)
    df_ = df_[cols].dropna(subset=dropna_cols)
    return df_.loc[~df_[dropna_cols].eq("").any(axis=1)]


def process_single_team(
    team_short_name: str,
    season: Season,
) -> list[dict[str, TeamGameweek]]:
    """

    Args:
    ----
        team_short_name: Team FPL API short name.
        season: Season.

    Returns:
    -------
        A list containing teams' gameweek data for the team.

    """
    folder_structure: Path = (
        DATA_FOLDER_FBREF / season.folder / "team_matchlogs" / team_short_name
    )

    df_team_gw: pd.DataFrame = reduce(
        lambda left, right: left.merge(
            right,
            on="date",
            how="left",
            validate="1:1",
        ),
        [
            process_single_stat(
                folder_structure,
                "schedule_for",
                {"opp_formation": "formation_vs"},
                [
                    "opponent",
                    "date",
                    "venue",
                    "result",
                    "possession",
                    "formation",
                    "formation_vs",
                ],
                ["date", "result"],
            ),
            process_single_stat(
                folder_structure,
                "shooting_for",
                {
                    "header_for_against_date": "date",
                    "header_standard_shots_on_target": "shots_on_target",
                    "header_standard_pens_att": "pens_won",
                    "header_standard_pens_made": "pens_scored",
                    "header_expected_npxg": "npxg",
                },
                ["date", "shots_on_target", "pens_won", "pens_scored", "npxg"],
                ["date"],
            ),
            process_single_stat(
                folder_structure,
                "shooting_against",
                {
                    "header_for_against_date": "date",
                    "header_standard_shots_on_target": "shots_on_target_vs",
                    "header_expected_npxg": "npxg_vs",
                },
                ["date", "shots_on_target_vs", "npxg_vs"],
                ["date"],
            ),
            process_single_stat(
                folder_structure,
                "passing_for",
                {
                    "header_for_against_date": "date",
                    "assisted_shots": "key_passes",
                },
                ["date", "key_passes", "pass_xa"],
                ["date"],
            ),
            process_single_stat(
                folder_structure,
                "gca_for",
                {
                    "header_for_against_date": "date",
                    "header_sca_types_sca": "sca",
                    "header_gca_types_gca": "gca",
                },
                ["date", "sca", "gca"],
                ["date"],
            ),
            process_single_stat(
                folder_structure,
                "defense_for",
                {
                    "header_for_against_date": "date",
                    "header_tackles_tackles_won": "tackles_won",
                    "header_blocks_blocks": "blocks",
                },
                [
                    "date",
                    "tackles_won",
                    "blocks",
                    "interceptions",
                    "clearances",
                ],
                ["date"],
            ),
            process_single_stat(
                folder_structure,
                "misc_for",
                {
                    "header_for_against_date": "date",
                    "header_performance_cards_yellow": "yellow_cards",
                    "header_performance_cards_red": "red_cards",
                    "header_performance_fouls": "fouls_conceded",
                    "header_performance_fouled": "fouls_won",
                    "header_performance_pens_conceded": "pens_conceded",
                },
                [
                    "date",
                    "yellow_cards",
                    "red_cards",
                    "fouls_conceded",
                    "fouls_won",
                    "pens_conceded",
                ],
                ["date"],
            ),
            process_single_stat(
                folder_structure,
                "misc_against",
                {
                    "header_for_against_date": "date",
                    "header_performance_cards_yellow": "yellow_cards_vs",
                    "header_performance_cards_red": "red_cards_vs",
                },
                [
                    "date",
                    "yellow_cards_vs",
                    "red_cards_vs",
                ],
                ["date"],
            ),
            process_single_stat(
                folder_structure,
                "keeper_for",
                {
                    "header_for_against_date": "date",
                    "header_performance_gk_saves": "gk_saves",
                },
                ["date", "gk_saves"],
                ["date"],
            ),
        ],
    )

    team: Team = next(el for el in _list_teams if el.short_name == team_short_name)
    df_team_gw["opponent"] = [
        {t.fbref_name: t for t in _list_teams}.get(t) for t in df_team_gw["opponent"]
    ]
    df_team_gw = df_team_gw.sort_values(by="date", ascending=True)
    return [
        TeamGameweek.model_validate(
            {"team": team, "season": season.fbref_long_name, **row},
        ).model_dump()
        for row in df_team_gw.to_dict(orient="records")
    ]


def save_aggregate_team_matchlogs(
    season: Literal[Seasons.SEASON_2324, Seasons.SEASON_2425],
) -> None:
    """

    Args:
    ----
        season: Season.

    """
    dfs: list[dict[str, TeamGameweek]] = []
    _teams: list[str] = get_fbref_teams(season.value)
    for team_name in rich.progress.track(_teams):
        team: Team = next(el for el in _list_teams if el.fbref_name == team_name)
        df_temp: list[dict[str, TeamGameweek]] = process_single_team(
            team.short_name,
            season.value,
        )
        dfs += df_temp
    fpath: Path = DATA_FOLDER_FBREF / season.value.folder / "team_matchlogs.json"
    save_json({"team_matchlogs": dfs}, fpath=fpath, default=str)
    logger.info(
        "Team matchlogs saved for all clubs from Season: {}",
        season.value.fbref_name,
    )


if __name__ == "__main__":
    # save_aggregate_team_matchlogs(Seasons.SEASON_2324)
    save_aggregate_team_matchlogs(Seasons.SEASON_2425)
