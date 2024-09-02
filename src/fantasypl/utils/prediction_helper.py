"""Helper functions for current season predictions."""

import json
import operator
from functools import reduce
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from pulp import LpBinary, LpProblem, LpVariable, value  # type: ignore[import-untyped]

from fantasypl.config.constants.folder_config import DATA_FOLDER_REF, MODEL_FOLDER
from fantasypl.config.constants.prediction_config import (
    MAX_DEF_COUNT,
    MAX_FWD_COUNT,
    MAX_GKP_COUNT,
    MAX_MID_COUNT,
    MAX_SAME_CLUB_COUNT,
    MIN_DEF_COUNT,
    MIN_FWD_COUNT,
    MIN_GKP_COUNT,
    MIN_MID_COUNT,
    TOTAL_DEF_COUNT,
    TOTAL_FWD_COUNT,
    TOTAL_GKP_COUNT,
    TOTAL_LINEUP_COUNT,
    TOTAL_MID_COUNT,
)
from fantasypl.config.models.player import Player
from fantasypl.config.models.team import Team


with Path.open(DATA_FOLDER_REF / "teams.json", "r") as f:
    list_teams: list[Team] = [
        Team.model_validate(el) for el in json.load(f).get("teams")
    ]

with Path.open(DATA_FOLDER_REF / "players.json", "r") as fl:
    list_players: list[Player] = [
        Player.model_validate(el) for el in json.load(fl).get("players")
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


def prepare_df_for_optimization(
    gameweek: int, weights_decays_base: list[float]
) -> pd.DataFrame:
    """

    Args:
    ----
        gameweek: Gameweek.
        weights_decays_base: Per GW decay for predicted points.

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
        gameweek + 1: weights_decays_base[0],
        gameweek + 2: weights_decays_base[1],
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


def prepare_common_lists_from_df(
    df_values: pd.DataFrame,
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.float32],
    npt.NDArray[np.int32],
    npt.NDArray[np.str_],
    npt.NDArray[np.str_],
]:
    """

    Args:
    ----
        df_values: The prepared dataframe for optimization.

    Returns:
    -------
        A tuple containing players, points, prices, positions and teams

    """
    players: npt.NDArray[np.int32] = df_values["code"].to_numpy()
    points: npt.NDArray[np.float32] = df_values["weighted_points"].to_numpy()
    prices: npt.NDArray[np.int32] = df_values["now_cost"].to_numpy()
    positions: npt.NDArray[np.str_] = df_values["fpl_position"].to_numpy()
    teams: npt.NDArray[np.str_] = df_values["team"].to_numpy()
    return players, points, prices, positions, teams


def prepare_lp_variables(
    players: npt.NDArray[np.int32],
) -> tuple[
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
]:
    """

    Args:
    ----
        players: The list of players.

    Returns:
    -------
        A tuple containing the LP variables for lineup and bench.

    """
    lineup: npt.NDArray[LpVariable] = np.array([
        LpVariable(f"l{pl}", cat=LpBinary) for pl in players
    ])
    bench_gk: npt.NDArray[LpVariable] = np.array([
        LpVariable(f"bg{pl}", cat=LpBinary) for pl in players
    ])
    bench_1: npt.NDArray[LpVariable] = np.array([
        LpVariable(f"bf{pl}", cat=LpBinary) for pl in players
    ])
    bench_2: npt.NDArray[LpVariable] = np.array([
        LpVariable(f"bs{pl}", cat=LpBinary) for pl in players
    ])
    bench_3: npt.NDArray[LpVariable] = np.array([
        LpVariable(f"bt{pl}", cat=LpBinary) for pl in players
    ])
    captain: npt.NDArray[LpVariable] = np.array([
        LpVariable(f"c{pl}", cat=LpBinary) for pl in players
    ])
    return lineup, bench_gk, bench_1, bench_2, bench_3, captain


def prepare_additional_lpvariables(
    players: npt.NDArray[np.int32],
) -> tuple[
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
]:
    """

    Args:
    ----
        players: The list of players.

    Returns:
    -------
        A tuple containing the LP variables for transfers.

    """
    initial_squad: npt.NDArray[LpVariable] = np.array([
        LpVariable(f"sq{pl}", cat=LpBinary) for pl in players
    ])
    transfers_out: npt.NDArray[LpVariable] = np.array([
        LpVariable(f"out{pl}", cat=LpBinary) for pl in players
    ])
    transfers_in_free: npt.NDArray[LpVariable] = np.array([
        LpVariable(f"ft{pl}", cat=LpBinary) for pl in players
    ])
    transfers_in_hit: npt.NDArray[LpVariable] = np.array([
        LpVariable(f"hit{pl}", cat=LpBinary) for pl in players
    ])
    return initial_squad, transfers_out, transfers_in_free, transfers_in_hit


def add_count_constraints(  # noqa: PLR0917
    problem: LpProblem,
    lineup: npt.NDArray[LpVariable],
    bench_gk: npt.NDArray[LpVariable],
    bench_1: npt.NDArray[LpVariable],
    bench_2: npt.NDArray[LpVariable],
    bench_3: npt.NDArray[LpVariable],
    captain: npt.NDArray[LpVariable],
) -> LpProblem:
    """

    Args:
    ----
        problem: The LP problem.
        lineup: The LP variable for lineup.
        bench_gk: The LP variable for the bench gk.
        bench_1: The LP variable for first bench.
        bench_2: The LP variable for second bench.
        bench_3: The LP variable for third bench.
        captain: The LP variable for captain.

    Returns:
    -------
        The LP problem with count constraints.

    """
    problem.addConstraint(sum(lineup) == TOTAL_LINEUP_COUNT)
    problem.addConstraint(sum(bench_gk) == 1)
    problem.addConstraint(sum(bench_1) == 1)
    problem.addConstraint(sum(bench_2) == 1)
    problem.addConstraint(sum(bench_3) == 1)
    problem.addConstraint(sum(captain) == 1)
    return problem


def add_position_constraints(  # noqa: PLR0917
    problem: LpProblem,
    mask: npt.NDArray[np.bool],
    lineup: npt.NDArray[LpVariable],
    bench: list[npt.NDArray[LpVariable]],
    min_count: int,
    max_count: int,
    total_count: int,
) -> LpProblem:
    """

    Args:
    ----
        problem: LP problem.
        mask: Position mask.
        lineup: Lineup array.
        bench: Bench arrays.
        min_count: Minimum count for position.
        max_count: Maximum count for position.
        total_count: Total count for position.

    Returns:
    -------
        The LP problem with positional constraints.

    """
    problem.addConstraint(mask @ lineup >= min_count)
    problem.addConstraint(mask @ lineup <= max_count)
    problem.addConstraint(mask @ (lineup + reduce(operator.add, bench)) == total_count)
    return problem


def add_other_constraints(  # noqa: PLR0917
    problem: LpProblem,
    lineup: npt.NDArray[LpVariable],
    bench_gk: npt.NDArray[LpVariable],
    bench_1: npt.NDArray[LpVariable],
    bench_2: npt.NDArray[LpVariable],
    bench_3: npt.NDArray[LpVariable],
    captain: npt.NDArray[LpVariable],
    positions: npt.NDArray[np.str_],
    teams: npt.NDArray[np.str_],
) -> LpProblem:
    """

    Args:
    ----
        problem: The LP problem.
        lineup: The LP variable for lineup.
        bench_gk: The LP variable for the bench gk.
        bench_1: The LP variable for first bench.
        bench_2: The LP variable for second bench.
        bench_3: The LP variable for third bench.
        captain: The LP variable for captain.
        positions: The positions array.
        teams: The teams array.

    Returns:
    -------
        The LP problem with sub-not-in-lineup, captain-in-lineup, positional
        and team constraints.

    """
    sub_not_in_lineup_expressions: npt.NDArray[LpVariable] = (
        lineup + bench_gk + bench_1 + bench_2 + bench_3
    )
    for expr in sub_not_in_lineup_expressions:
        problem.addConstraint(expr <= 1)
    capt_in_lineup_expressions: npt.NDArray[LpVariable] = lineup - captain
    for expr in capt_in_lineup_expressions:
        problem.addConstraint(expr >= 0)

    problem = add_position_constraints(
        problem,
        np.array(positions == "GKP"),
        lineup,
        [bench_gk],
        MIN_GKP_COUNT,
        MAX_GKP_COUNT,
        TOTAL_GKP_COUNT,
    )
    problem = add_position_constraints(
        problem,
        np.array(positions == "DEF"),
        lineup,
        [bench_1, bench_2, bench_3],
        MIN_DEF_COUNT,
        MAX_DEF_COUNT,
        TOTAL_DEF_COUNT,
    )
    problem = add_position_constraints(
        problem,
        np.array(positions == "MID"),
        lineup,
        [bench_1, bench_2, bench_3],
        MIN_MID_COUNT,
        MAX_MID_COUNT,
        TOTAL_MID_COUNT,
    )
    problem = add_position_constraints(
        problem,
        np.array(positions == "FWD"),
        lineup,
        [bench_1, bench_2, bench_3],
        MIN_FWD_COUNT,
        MAX_FWD_COUNT,
        TOTAL_FWD_COUNT,
    )

    for club in np.unique(teams):
        club_mask: npt.NDArray[np.bool] = np.array(teams == club)
        problem.addConstraint(
            club_mask @ (lineup + bench_gk + bench_1 + bench_2 + bench_3)
            <= MAX_SAME_CLUB_COUNT
        )

    return problem


def arrange_return_and_log_variables(  # noqa: PLR0917
    problem: LpProblem,
    lineup: npt.NDArray[LpVariable],
    bench_gk: npt.NDArray[LpVariable],
    bench_1: npt.NDArray[LpVariable],
    bench_2: npt.NDArray[LpVariable],
    bench_3: npt.NDArray[LpVariable],
) -> tuple[
    npt.NDArray[np.float32],
    list[str],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    list[str],
    str,
]:
    """

    Args:
    ----
        problem: The LP problem.
        lineup: The LP variable for lineup.
        bench_gk: The LP variable for the bench gk.
        bench_1: The LP variable for first bench.
        bench_2: The LP variable for second bench.
        bench_3: The LP variable for third bench.

    Returns:
    -------
        A tuple containing all variables required for return and logging purposes.

    """
    optimal_lineup: npt.NDArray[np.float32] = np.array([value(var) for var in lineup])
    optimal_bench_gk: npt.NDArray[np.float32] = np.array([
        value(var) for var in bench_gk
    ])
    optimal_bench_1: npt.NDArray[np.float32] = np.array([value(var) for var in bench_1])
    optimal_bench_2: npt.NDArray[np.float32] = np.array([value(var) for var in bench_2])
    optimal_bench_3: npt.NDArray[np.float32] = np.array([value(var) for var in bench_3])
    selected_players: list[str] = [
        v.name for v in problem.variables() if v.varValue == 1
    ]
    lineup_players: list[str] = [
        el.fpl_web_name for el in list_players if f"l{el.fpl_code}" in selected_players
    ]
    bench_players: list[str] = (
        [
            el.fpl_web_name
            for el in list_players
            if f"bg{el.fpl_code}" in selected_players
        ]
        + [
            el.fpl_web_name
            for el in list_players
            if f"bf{el.fpl_code}" in selected_players
        ]
        + [
            el.fpl_web_name
            for el in list_players
            if f"bs{el.fpl_code}" in selected_players
        ]
        + [
            el.fpl_web_name
            for el in list_players
            if f"bt{el.fpl_code}" in selected_players
        ]
    )
    captain_player: str = next(
        el.fpl_web_name for el in list_players if f"c{el.fpl_code}" in selected_players
    )
    return (
        optimal_lineup,
        lineup_players,
        optimal_bench_gk,
        optimal_bench_1,
        optimal_bench_2,
        optimal_bench_3,
        bench_players,
        captain_player,
    )
