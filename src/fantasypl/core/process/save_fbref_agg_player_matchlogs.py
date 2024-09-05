"""Functions for creating player matchlogs for entire season."""

import os
from functools import reduce
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import rich.progress
from loguru import logger

from fantasypl.config.constants import (
    DATA_FOLDER_FBREF,
    FBREF_POSITION_MAPPING,
)
from fantasypl.config.schemas import (
    PlayerGameWeek,
    Season,
    Seasons,
    Team,
)
from fantasypl.utils import (
    get_fbref_teams,
    get_list_players,
    get_list_teams,
    get_team_gameweek_json_to_df,
    save_json,
)


if TYPE_CHECKING:
    from pathlib import Path


def filter_minutes(group: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    group
        The grouped data to filter.

    Returns
    -------
        A pandas dataframe clipped by the first and last nonzero minutes

    """
    first_nonzero_idx: int = group["minutes"].ne(0).idxmax()  # type: ignore[assignment]
    last_nonzero_idx: int = group["minutes"].iloc[::-1].ne(0).idxmax()  # type: ignore[assignment]
    return group.loc[first_nonzero_idx:last_nonzero_idx]


def process_single_team(  # noqa: PLR0914, PLR0915
    team: Team, season: Season
) -> list[dict[str, PlayerGameWeek]]:
    """

    Parameters
    ----------
    team
        A single Team object.
    season
        The season under process.

    Returns
    -------
        A list containing all players' gameweek data for the team.

    """
    list_files: list[str] = next(
        iter(
            os.walk(
                DATA_FOLDER_FBREF / season.folder / "matches" / team.short_name
            )
        )
    )[2]
    dfs_summary: list[pd.DataFrame] = []
    dfs_passing: list[pd.DataFrame] = []
    dfs_defense: list[pd.DataFrame] = []
    dfs_misc: list[pd.DataFrame] = []
    dfs_keeper: list[pd.DataFrame] = []

    for fl in list_files:
        df_stats: pd.DataFrame = pd.read_csv(
            DATA_FOLDER_FBREF
            / season.folder
            / "matches"
            / team.short_name
            / fl
        )
        df_stats["starts"] = np.where(
            df_stats["player"].str.contains("\xa0"), 0, 1
        )
        df_stats["player"] = df_stats["player"].str.strip()
        _join_cols: list[str] = ["player", "date", "venue"]
        match fl:
            case fl if "summary" in fl:
                df_stats["short_position"] = (
                    df_stats["position"]
                    .str.split(",")
                    .str[0]
                    .map(FBREF_POSITION_MAPPING)
                )
                df_stats = df_stats.rename(
                    columns={
                        "header_performance_shots_on_target": "shots_on_target",  # noqa: E501
                        "header_performance_cards_yellow": "yellow_cards",
                        "header_performance_cards_red": "red_cards",
                        "header_performance_pens_att": "pens_taken",
                        "header_performance_pens_made": "pens_scored",
                        "header_expected_npxg": "npxg",
                        "header_expected_xg_assist": "xa",
                        "header_sca_sca": "sca",
                        "header_sca_gca": "gca",
                        "header_carries_progressive_carries": "progressive_carries",  # noqa: E501
                    }
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
                df_stats = df_stats.rename(
                    columns={"assisted_shots": "key_passes"}
                )
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
                    }
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
                    }
                )
                df_stats = df_stats[[*_join_cols, "gk_saves", "gk_psxg"]]
                dfs_keeper.append(df_stats)
            case _:
                logger.error("Untracked file: {}", fl)

    df_summary: pd.DataFrame = (
        pd.concat(dfs_summary, ignore_index=True)
        if dfs_summary
        else pd.DataFrame()
    )
    df_passing: pd.DataFrame = (
        pd.concat(dfs_passing, ignore_index=True)
        if dfs_passing
        else pd.DataFrame()
    )
    df_defense: pd.DataFrame = (
        pd.concat(dfs_defense, ignore_index=True)
        if dfs_defense
        else pd.DataFrame()
    )
    df_misc: pd.DataFrame = (
        pd.concat(dfs_misc, ignore_index=True) if dfs_misc else pd.DataFrame()
    )
    df_keeper: pd.DataFrame = (
        pd.concat(dfs_keeper, ignore_index=True)
        if dfs_keeper
        else pd.DataFrame()
    )

    df_final: pd.DataFrame = reduce(
        lambda left, right: left.merge(
            right, on=_join_cols, how="left", validate="m:m"
        ),
        [df_summary, df_passing, df_defense, df_misc, df_keeper],
    )
    df_team_gw: pd.DataFrame = get_team_gameweek_json_to_df(season)
    df_team_gw["date"] = df_team_gw["date"].astype(str)
    df_dates: pd.DataFrame = df_team_gw.loc[
        df_team_gw["team"] == team, ["date", "venue"]
    ]
    df_ids: pd.DataFrame = pd.DataFrame({
        "player": df_final["player"].unique()
    })
    df_dates = df_ids.merge(df_dates, how="cross")
    df_dates["date"] = df_dates["date"].astype(str)
    df_final = df_final.merge(
        df_dates, on=["player", "date", "venue"], how="right", validate="1:1"
    )
    df_final.loc[:, df_final.columns != "short_position"] = df_final.loc[
        :, df_final.columns != "short_position"
    ].fillna(0)
    df_final[["short_position"]] = df_final[["short_position"]].map(
        lambda x: None if pd.isna(x) else x
    )
    df_final = (
        df_final.groupby("player")
        .apply(filter_minutes, include_groups=False)
        .reset_index(level="player")
    )
    df_final["player"] = [
        {p.fbref_name: p for p in get_list_players()}.get(p)
        for p in df_final["player"]
    ]
    return [
        PlayerGameWeek.model_validate({
            "team": team,
            "season": season.fbref_long_name,
            **row,
        }).model_dump()
        for row in df_final.to_dict(orient="records")
    ]


def save_aggregate_player_matchlogs(
    season: Literal[Seasons.SEASON_2324, Seasons.SEASON_2425],  # type: ignore[valid-type]
) -> None:
    """

    Parameters
    ----------
    season
        The season under process.

    """
    dfs: list[dict[str, PlayerGameWeek]] = []
    _teams: list[str] = get_fbref_teams(season.value)
    for team_name in rich.progress.track(_teams):
        team: Team = next(
            el for el in get_list_teams() if el.fbref_name == team_name
        )
        df_temp: list[dict[str, PlayerGameWeek]] = process_single_team(
            team, season.value
        )
        dfs += df_temp
    fpath: Path = (
        DATA_FOLDER_FBREF / season.value.folder / "player_matchlogs.json"
    )
    save_json({"player_matchlogs": dfs}, fpath=fpath, default=str)
    logger.info(
        "Player matchlogs saved for all clubs from Season: {}",
        season.value.fbref_name,
    )


if __name__ == "__main__":
    # save_aggregate_player_matchlogs(Seasons.SEASON_2324)
    save_aggregate_player_matchlogs(Seasons.SEASON_2425)
