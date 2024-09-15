"""Functions for creating team matchlogs for entire season."""

from functools import reduce
from pathlib import Path
from typing import Literal

import pandas as pd
import rich.progress
from loguru import logger

from fantasypl.config.constants import (
    DATA_FOLDER_FBREF,
)
from fantasypl.config.schemas import Season, Seasons, Team, TeamGameweek
from fantasypl.utils import get_fbref_teams, get_list_teams, save_json


def process_single_stat(
    folder_structure: Path,
    stat: str,
    rename_dict: dict[str, str],
    cols: list[str],
    dropna_cols: list[str],
) -> pd.DataFrame:
    """
    Process team gameweeks data for a single stat.

    Parameters
    ----------
    folder_structure
        Path for the parent folder.
    stat
        File name to look at.
    rename_dict
        Columns rename dictionary.
    cols
        Columns to be selected.
    dropna_cols
        Columns to mark empty rows.

    Returns
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
    Return team gameweeks data for a single team.

    Parameters
    ----------
    team_short_name
        Team FPL API short name.
    season
        The season under process.

    Returns
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
                {},
                [
                    "opponent",
                    "date",
                    "venue",
                    "result",
                    "possession",
                ],
                ["date", "result"],
            ),
            process_single_stat(
                folder_structure,
                "shooting_for",
                {
                    "header_for_against_date": "date",
                    "header_standard_shots": "shots",
                    "header_standard_shots_on_target": "shots_on_target",
                    "header_standard_average_shot_distance": "average_shot_distance",  # noqa: E501
                    "header_standard_pens_att": "pens_won",
                    "header_standard_pens_made": "pens_scored",
                    "header_expected_npxg": "npxg",
                },
                [
                    "date",
                    "shots",
                    "shots_on_target",
                    "average_shot_distance",
                    "npxg",
                    "pens_won",
                    "pens_scored",
                ],
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
                    "header_passes_total_passes_completed": "passes_completed",
                    "assisted_shots": "key_passes",
                },
                [
                    "date",
                    "passes_completed",
                    "progressive_passes",
                    "key_passes",
                    "pass_xa",
                    "passes_into_final_third",
                ],
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
                "gca_for",
                {
                    "header_for_against_date": "date",
                    "header_sca_types_sca": "sca_vs",
                    "header_gca_types_gca": "gca_vs",
                },
                ["date", "sca_vs", "gca_vs"],
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
                "possession_for",
                {
                    "header_for_against_date": "date",
                    "header_carries_progressive_carries": "progressive_carries",  # noqa: E501
                },
                [
                    "date",
                    "progressive_carries",
                ],
                ["date"],
            ),
            process_single_stat(
                folder_structure,
                "misc_for",
                {
                    "header_for_against_date": "date",
                    "header_performance_ball_recoveries": "ball_recoveries",
                    "header_aerials_aerials_won_pct": "aerials_won_pct",
                    "header_performance_cards_yellow": "yellow_cards",
                    "header_performance_cards_red": "red_cards",
                    "header_performance_fouls": "fouls_conceded",
                    "header_performance_fouled": "fouls_won",
                    "header_performance_pens_conceded": "pens_conceded",
                },
                [
                    "date",
                    "ball_recoveries",
                    "aerials_won_pct",
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
                ["date", "yellow_cards_vs", "red_cards_vs"],
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

    team: Team = next(
        el for el in get_list_teams() if el.short_name == team_short_name
    )
    df_team_gw["opponent"] = [
        {t.fbref_name: t for t in get_list_teams()}.get(t)
        for t in df_team_gw["opponent"]
    ]
    df_team_gw = df_team_gw.sort_values(by="date", ascending=True)
    return [
        TeamGameweek.model_validate({
            "team": team,
            "season": season.fbref_long_name,
            **row,
        }).model_dump()
        for row in df_team_gw.to_dict(orient="records")
    ]


def save_aggregate_team_matchlogs(
    season: Literal[Seasons.SEASON_2324, Seasons.SEASON_2425],
) -> None:
    """
    Return all team gameweeks data.

    Parameters
    ----------
    season
        The season under process.

    """
    dfs: list[dict[str, TeamGameweek]] = []
    _teams: list[str] = get_fbref_teams(season.value)
    for team_name in rich.progress.track(_teams):
        team: Team = next(
            el for el in get_list_teams() if el.fbref_name == team_name
        )
        df_temp: list[dict[str, TeamGameweek]] = process_single_team(
            team.short_name,
            season.value,
        )
        dfs += df_temp
    fpath: Path = (
        DATA_FOLDER_FBREF / season.value.folder / "team_matchlogs.json"
    )
    save_json({"team_matchlogs": dfs}, fpath=fpath, default=str)
    logger.info(
        "Team matchlogs saved for all clubs from Season: {}",
        season.value.fbref_name,
    )


if __name__ == "__main__":
    save_aggregate_team_matchlogs(Seasons.SEASON_2324)
    save_aggregate_team_matchlogs(Seasons.SEASON_2425)
