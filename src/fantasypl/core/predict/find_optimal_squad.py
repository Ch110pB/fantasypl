"""Functions to find the optimal FPL squad."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pulp  # type: ignore[import-untyped]
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_REF, MODEL_FOLDER
from fantasypl.config.constants.prediction_config import (
    BENCH_WEIGHTS_ARRAY,
    WEIGHTS_DECAYS_BASE,
)
from fantasypl.config.models.player import Player
from fantasypl.utils.prediction_helper import (
    add_count_constraints,
    add_other_constraints,
    arrange_return_and_log_variables,
    prepare_common_lists_from_df,
    prepare_df_for_optimization,
    prepare_lp_variables,
)


if TYPE_CHECKING:
    import pandas as pd


with Path.open(DATA_FOLDER_REF / "players.json", "r") as fl:
    _list_players: list[Player] = [
        Player.model_validate(el) for el in json.load(fl).get("players")
    ]


def find_squad(  # noqa: PLR0914
    gameweek: int,
    budget: int,
    bench_weights: list[float] | None = None,
    weights_decays_base: list[float] | None = None,
) -> tuple[list[str], list[str], str]:
    """

    Args:
    ----
        gameweek: Gameweek.
        budget: Total budget available.
        bench_weights: Weights given to points of bench players.
        weights_decays_base: Per GW decay for predicted points.

    Returns:
    -------
        A tuple containing the lineup, bench and the captain.

    """
    if weights_decays_base is None:
        weights_decays_base = WEIGHTS_DECAYS_BASE
    if bench_weights is None:
        bench_weights = BENCH_WEIGHTS_ARRAY

    df_values: pd.DataFrame = prepare_df_for_optimization(gameweek, weights_decays_base)
    players, points, prices, positions, teams = prepare_common_lists_from_df(df_values)

    problem: pulp.LpProblem = pulp.LpProblem("squad_building", pulp.LpMaximize)
    lineup, bench_gk, bench_1, bench_2, bench_3, captain = prepare_lp_variables(players)

    problem.setObjective(
        points @ (lineup + captain)
        + (bench_weights[0] * points) @ bench_gk
        + (bench_weights[1] * points) @ bench_1
        + (bench_weights[2] * points) @ bench_2
        + (bench_weights[3] * points) @ bench_3
    )

    problem = add_count_constraints(
        problem, lineup, bench_gk, bench_1, bench_2, bench_3, captain
    )
    problem.addConstraint(
        prices @ (lineup + bench_gk + bench_1 + bench_2 + bench_3) <= budget
    )
    problem = add_other_constraints(
        problem, lineup, bench_gk, bench_1, bench_2, bench_3, captain, positions, teams
    )

    problem.writeLP(
        f"{MODEL_FOLDER}/predictions/player/gameweek_{gameweek}/{problem.name}.lp"
    )
    problem.solve()
    (
        optimal_lineup,
        lineup_players,
        optimal_bench_gk,
        optimal_bench_1,
        optimal_bench_2,
        optimal_bench_3,
        bench_players,
        captain_player,
    ) = arrange_return_and_log_variables(
        problem, lineup, bench_gk, bench_1, bench_2, bench_3
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
    eleven, subs, cap = find_squad(4, 1000)
    logger.info("Starting Lineup: {}", eleven)
    logger.info("Bench: {}", subs)
    logger.info("Captain: {}", cap)
