"""Functions to find the optimal FPL squad."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pulp  # type: ignore[import-untyped]
from loguru import logger

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
    TOTAL_BENCH_COUNT,
    TOTAL_DEF_COUNT,
    TOTAL_FWD_COUNT,
    TOTAL_GKP_COUNT,
    TOTAL_LINEUP_COUNT,
    TOTAL_MID_COUNT,
)
from fantasypl.config.models.player import Player


with Path.open(DATA_FOLDER_REF / "players.json", "r") as fl:
    _list_players: list[Player] = [
        Player.model_validate(el) for el in json.load(fl).get("players")
    ]


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
        ["code", "fpl_position", "team", "gameweek", "now_cost", "points"]
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
        ])["weighted_points"]
        .sum()
        .reset_index()
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


def find_squad(  # noqa: PLR0914
    gameweek: int, budget: int, bench_weight: float = 0.21
) -> tuple[list[str], list[str], str]:
    """

    Args:
    ----
        gameweek: Gameweek.
        budget: Total budget available.
        bench_weight: Weight given to points of bench players.

    Returns:
    -------
        A tuple containing the lineup, bench and the captain.

    """
    df_values: pd.DataFrame = prepare_df_for_optimization(gameweek)
    players: npt.NDArray[np.int32] = df_values["code"].to_numpy()
    points: npt.NDArray[np.float32] = df_values["weighted_points"].to_numpy()
    prices: npt.NDArray[np.int32] = df_values["now_cost"].to_numpy()
    positions: npt.NDArray[np.str_] = df_values["fpl_position"].to_numpy()
    teams: npt.NDArray[np.str_] = df_values["team"].to_numpy()

    problem: pulp.LpProblem = pulp.LpProblem("Squad_Building", pulp.LpMaximize)

    lineup: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"l{pl}", cat=pulp.LpBinary) for pl in players
    ])
    bench: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"b{pl}", cat=pulp.LpBinary) for pl in players
    ])
    captain: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"c{pl}", cat=pulp.LpBinary) for pl in players
    ])

    problem.setObjective(points @ (lineup + captain) + (bench_weight * points) @ bench)

    problem.addConstraint(sum(lineup) == TOTAL_LINEUP_COUNT)
    problem.addConstraint(sum(bench) == TOTAL_BENCH_COUNT)
    problem.addConstraint(sum(captain) == 1)

    problem.addConstraint(prices @ (lineup + bench) <= budget)

    sub_not_in_lineup_expressions: npt.NDArray[pulp.LpVariable] = lineup + bench
    for expr in sub_not_in_lineup_expressions:
        problem.addConstraint(expr <= 1)
    capt_in_lineup_expressions: npt.NDArray[pulp.LpVariable] = lineup - captain
    for expr in capt_in_lineup_expressions:
        problem.addConstraint(expr >= 0)

    problem = add_position_constraints(
        problem,
        np.array(positions == "GKP"),
        lineup,
        bench,
        MIN_GKP_COUNT,
        MAX_GKP_COUNT,
        TOTAL_GKP_COUNT,
    )
    problem = add_position_constraints(
        problem,
        np.array(positions == "DEF"),
        lineup,
        bench,
        MIN_DEF_COUNT,
        MAX_DEF_COUNT,
        TOTAL_DEF_COUNT,
    )
    problem = add_position_constraints(
        problem,
        np.array(positions == "MID"),
        lineup,
        bench,
        MIN_MID_COUNT,
        MAX_MID_COUNT,
        TOTAL_MID_COUNT,
    )
    problem = add_position_constraints(
        problem,
        np.array(positions == "FWD"),
        lineup,
        bench,
        MIN_FWD_COUNT,
        MAX_FWD_COUNT,
        TOTAL_FWD_COUNT,
    )

    for club in df_values["team"].unique():
        club_mask: npt.NDArray[np.bool] = np.array(teams == club)
        problem.addConstraint(club_mask @ (lineup + bench) <= MAX_SAME_CLUB_COUNT)

    problem.solve()
    optimal_lineup: npt.NDArray[np.float32] = np.array([
        pulp.value(var) for var in lineup
    ])
    optimal_bench: npt.NDArray[np.float32] = np.array([
        pulp.value(var) for var in bench
    ])
    selected_players: list[str] = [
        v.name for v in problem.variables() if v.varValue == 1
    ]
    lineup_players: list[str] = [
        el.fpl_web_name for el in _list_players if f"l{el.fpl_code}" in selected_players
    ]
    bench_players: list[str] = [
        el.fpl_web_name for el in _list_players if f"b{el.fpl_code}" in selected_players
    ]
    captain_player: str = next(
        el.fpl_web_name for el in _list_players if f"c{el.fpl_code}" in selected_players
    )
    logger.info("Optimization complete for fresh squad.")
    logger.info("Predicted Lineup Points: {}", optimal_lineup @ points)
    logger.info("Total Cost: {}", optimal_lineup @ prices + optimal_bench @ prices)
    return lineup_players, bench_players, captain_player


if __name__ == "__main__":
    eleven, subs, cap = find_squad(3, 1000)
    logger.info("Starting Lineup: {}", eleven)
    logger.info("Bench: {}", subs)
    logger.info("Captain: {}", cap)
