"""Helper functions for current season predictions."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pulp  # type: ignore[import-untyped]

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


def prepare_df_for_optimization(gameweek: int) -> pd.DataFrame:
    """

    Args:
    ----
        gameweek: Gameweek.

    Returns:
    -------
        A dataframe containing the features for the optimization problem.

    """
    df_expected_points: pd.DataFrame = pd.read_csv(
        MODEL_FOLDER
        / "predictions/player"
        / f"gameweek_{gameweek}"
        / "prediction_xpoints.csv"
    )
    df_expected_points = df_expected_points[
        [
            "code",
            "fpl_position",
            "team",
            "gameweek",
            "now_cost",
            "points",
            "selected_by_percent",
        ]
    ]
    df_expected_points["weighted_points"] = df_expected_points[
        "points"
    ] * df_expected_points["gameweek"].map({
        gameweek: 1.0,
        gameweek + 1: 0.84,
        gameweek + 2: 0.84**2,
    })
    df_values = (
        df_expected_points.groupby([
            "code",
            "team",
            "fpl_position",
            "now_cost",
            "selected_by_percent",
        ])["weighted_points"]
        .sum()
        .reset_index()
    )
    df_values["weighted_points"] = df_values["weighted_points"].round(4) + (
        (100 - df_values["selected_by_percent"]) * 1e-6
    )
    return df_values.sort_values(by="code", ascending=True)


def add_position_constraints(  # noqa: PLR0917
    problem: pulp.LpProblem,
    mask: npt.NDArray[np.bool],
    lineup: npt.NDArray[pulp.LpVariable],
    bench: npt.NDArray[pulp.LpVariable],
    min_count: int,
    max_count: int,
    total_count: int,
) -> pulp.LpProblem:
    """

    Args:
    ----
        problem: LP problem.
        mask: Position mask.
        lineup: Lineup array.
        bench: Bench array.
        min_count: Minimum count for position.
        max_count: Maximum count for position.
        total_count: Total count for position.

    Returns:
    -------
        The LP problem with positional constraints.

    """
    problem.addConstraint(mask @ lineup >= min_count)
    problem.addConstraint(mask @ lineup <= max_count)
    problem.addConstraint(mask @ (lineup + bench) == total_count)
    return problem
