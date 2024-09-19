"""Functions to get the matches in the next gameweek(s) to predict for."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from fantasypl.config.constants import (
    DATA_FOLDER_FPL,
    MODEL_FOLDER,
)
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import get_list_teams, save_pandas


def get_gw_matches(season: Season, gameweek: int) -> None:
    """
    Get the matches of the gameweek.

    Parameters
    ----------
    season
        The season under process.
    gameweek
        The gameweek under process.

    """
    with Path.open(
        DATA_FOLDER_FPL / season.folder / "fixtures.json",
        "r",
    ) as fl:
        list_fixtures: list[dict[str, Any]] = json.load(fl)
    df_fixtures: pd.DataFrame = pd.DataFrame(list_fixtures)[
        ["code", "event", "team_h", "team_a"]
    ]
    df_teams: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FPL / season.folder / "teams.csv",
    )
    teams_dict: dict[str, str] = {
        el[0]: el[1]
        for el in df_teams[["id", "code"]].to_dict(orient="split")["data"]
    }
    df_fixtures["team_h"] = [
        next(
            el.fbref_id
            for el in get_list_teams()
            if el.fpl_code == int(teams_dict[team_h])
        )
        for team_h in df_fixtures["team_h"]
    ]
    df_fixtures["team_a"] = [
        next(
            el.fbref_id
            for el in get_list_teams()
            if el.fpl_code == int(teams_dict[team_a])
        )
        for team_a in df_fixtures["team_a"]
    ]

    mask: pd.Series[bool] = (gameweek <= df_fixtures["event"].astype(int)) & (
        df_fixtures["event"].astype(int) <= gameweek + 2
    )
    df_gameweek: pd.DataFrame = df_fixtures.loc[mask, :]

    df_matches: pd.DataFrame = (
        (
            pd.concat(
                [
                    df_gameweek.assign(
                        team=df_gameweek["team_h"],
                        opponent=df_gameweek["team_a"],
                        gameweek=df_gameweek["event"].astype(int),
                        venue="Home",
                    ),
                    df_gameweek.assign(
                        team=df_gameweek["team_a"],
                        opponent=df_gameweek["team_h"],
                        gameweek=df_gameweek["event"].astype(int),
                        venue="Away",
                    ),
                ],
                ignore_index=True,
            )
            .drop(columns=["team_h", "team_a", "event"])
            .rename(columns={"code": "match_code"})
        )
        if not df_gameweek.empty
        else pd.DataFrame()
    )
    fpath: Path = (
        MODEL_FOLDER
        / "predictions/team"
        / f"gameweek_{gameweek}"
        / "fixtures.csv"
    )
    save_pandas(df=df_matches, fpath=fpath)
    logger.info("Fixtures saved for gameweek {}", gameweek)


if __name__ == "__main__":
    gw: int = 5
    this_season: Season = Seasons.SEASON_2425.value
    get_gw_matches(this_season, gw)
