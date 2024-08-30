"""Helper functions for current season predictions."""

import json
from pathlib import Path

import pandas as pd

from fantasypl.config.constants.folder_config import DATA_FOLDER_REF, MODEL_FOLDER
from fantasypl.config.models.team import Team


with Path.open(DATA_FOLDER_REF / "teams.json", "r") as f:
    list_teams: list[Team] = [
        Team.model_validate(el) for el in json.load(f).get("teams")
    ]


def process_gameweek_data(gameweek: int) -> pd.DataFrame:
    """

    Args:
    ----
        gameweek: Gameweek

    Returns:
    -------
        A dataframe containing gameweek fixtures with FBRef IDs for teams.

    """
    df_gameweek: pd.DataFrame = pd.read_csv(
        MODEL_FOLDER / "predictions/team" / f"gameweek_{gameweek}/fixtures.csv"
    )
    df_gameweek["team"] = [
        next(el.fbref_id for el in list_teams if el.fbref_id == x)
        for x in df_gameweek["team"]
    ]
    df_gameweek["opponent"] = [
        next(el.fbref_id for el in list_teams if el.fbref_id == x)
        for x in df_gameweek["opponent"]
    ]
    return df_gameweek


def pad_lists(row: pd.Series, df2: pd.DataFrame, col: str, group_col: str) -> pd.Series:  # type: ignore[type-arg]
    """

    Args:
    ----
        row: A row of running dataframe.
        df2: The aggregate dataframe.
        col: Column name.
        group_col: Group by column name.

    Returns:
    -------
        The modified row.

    """
    prev_season_value: str | int | float = df2.at[row[group_col], col]  # noqa: PD008
    if len(row[col]) < 5:  # noqa: PLR2004
        row[col] = [prev_season_value] * (5 - len(row[col])) + row[col]
    return row[col]  # type: ignore[no-any-return]
