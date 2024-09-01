"""Functions to find the optimal FPL squad."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
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
    TOTAL_DEF_COUNT,
    TOTAL_FWD_COUNT,
    TOTAL_GKP_COUNT,
    TOTAL_LINEUP_COUNT,
    TOTAL_MID_COUNT,
)
from fantasypl.config.models.player import Player
from fantasypl.utils.prediction_helper import (
    add_position_constraints,
    prepare_df_for_optimization,
)


if TYPE_CHECKING:
    import pandas as pd


with Path.open(DATA_FOLDER_REF / "players.json", "r") as fl:
    _list_players: list[Player] = [
        Player.model_validate(el) for el in json.load(fl).get("players")
    ]


# noinspection DuplicatedCode
def find_squad(  # noqa: PLR0914
    gameweek: int, budget: int, bench_weights: list[float] | None = None
) -> tuple[list[str], list[str], str]:
    """

    Args:
    ----
        gameweek: Gameweek.
        budget: Total budget available.
        bench_weights: Weights given to points of bench players.

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

    problem: pulp.LpProblem = pulp.LpProblem("squad_building", pulp.LpMaximize)

    lineup: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"l{pl}", cat=pulp.LpBinary) for pl in players
    ])
    bench_gk: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"bg{pl}", cat=pulp.LpBinary) for pl in players
    ])
    bench_1: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"bf{pl}", cat=pulp.LpBinary) for pl in players
    ])
    bench_2: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"bs{pl}", cat=pulp.LpBinary) for pl in players
    ])
    bench_3: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"bt{pl}", cat=pulp.LpBinary) for pl in players
    ])
    captain: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"c{pl}", cat=pulp.LpBinary) for pl in players
    ])

    if bench_weights is None:
        bench_weights = [0.03, 0.21, 0.1, 0.002]

    problem.setObjective(
        points @ (lineup + captain)
        + (bench_weights[0] * points) @ bench_gk
        + (bench_weights[1] * points) @ bench_1
        + (bench_weights[2] * points) @ bench_2
        + (bench_weights[3] * points) @ bench_3
    )

    problem.addConstraint(sum(lineup) == TOTAL_LINEUP_COUNT)
    problem.addConstraint(sum(bench_gk) == 1)
    problem.addConstraint(sum(bench_1) == 1)
    problem.addConstraint(sum(bench_2) == 1)
    problem.addConstraint(sum(bench_3) == 1)
    problem.addConstraint(sum(captain) == 1)

    problem.addConstraint(
        prices @ (lineup + bench_gk + bench_1 + bench_2 + bench_3) <= budget
    )

    sub_not_in_lineup_expressions: npt.NDArray[pulp.LpVariable] = (
        lineup + bench_gk + bench_1 + bench_2 + bench_3
    )
    for expr in sub_not_in_lineup_expressions:
        problem.addConstraint(expr <= 1)
    capt_in_lineup_expressions: npt.NDArray[pulp.LpVariable] = lineup - captain
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

    problem.writeLP(
        f"{MODEL_FOLDER}/predictions/player/gameweek_{gameweek}/{problem.name}.lp"
    )
    problem.solve()
    optimal_lineup: npt.NDArray[np.float32] = np.array([
        pulp.value(var) for var in lineup
    ])
    optimal_bench_gk: npt.NDArray[np.float32] = np.array([
        pulp.value(var) for var in bench_gk
    ])
    optimal_bench_1: npt.NDArray[np.float32] = np.array([
        pulp.value(var) for var in bench_1
    ])
    optimal_bench_2: npt.NDArray[np.float32] = np.array([
        pulp.value(var) for var in bench_2
    ])
    optimal_bench_3: npt.NDArray[np.float32] = np.array([
        pulp.value(var) for var in bench_3
    ])
    selected_players: list[str] = [
        v.name for v in problem.variables() if v.varValue == 1
    ]
    lineup_players: list[str] = [
        el.fpl_web_name for el in _list_players if f"l{el.fpl_code}" in selected_players
    ]
    bench_players: list[str] = (
        [
            el.fpl_web_name
            for el in _list_players
            if f"bg{el.fpl_code}" in selected_players
        ]
        + [
            el.fpl_web_name
            for el in _list_players
            if f"bf{el.fpl_code}" in selected_players
        ]
        + [
            el.fpl_web_name
            for el in _list_players
            if f"bs{el.fpl_code}" in selected_players
        ]
        + [
            el.fpl_web_name
            for el in _list_players
            if f"bt{el.fpl_code}" in selected_players
        ]
    )
    captain_player: str = next(
        el.fpl_web_name for el in _list_players if f"c{el.fpl_code}" in selected_players
    )
    logger.info("Optimization complete for fresh squad.")
    logger.info("Predicted Lineup Points: {}", optimal_lineup @ points)
    logger.info(
        "Total Cost: {}",
        (
            optimal_lineup
            + optimal_bench_gk
            + optimal_bench_1
            + optimal_bench_2
            + optimal_bench_3
        )
        @ prices,
    )
    return lineup_players, bench_players, captain_player


if __name__ == "__main__":
    eleven, subs, cap = find_squad(3, 1000)
    logger.info("Starting Lineup: {}", eleven)
    logger.info("Bench: {}", subs)
    logger.info("Captain: {}", cap)
