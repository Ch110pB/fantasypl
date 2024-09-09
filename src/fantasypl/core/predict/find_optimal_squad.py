"""Functions to find the optimal FPL squad."""

from typing import TYPE_CHECKING

import pulp  # type: ignore[import-untyped]
from loguru import logger

from fantasypl.config.constants import (
    BENCH_WEIGHTS_ARRAY,
    MODEL_FOLDER,
    WEIGHTS_DECAYS_BASE,
)
from fantasypl.config.schemas import Season, Seasons
from fantasypl.utils import (
    add_count_constraints,
    add_other_constraints,
    build_fpl_lineup,
    prepare_common_lists_from_df,
    prepare_df_for_optimization,
    prepare_essential_lp_variables,
    prepare_pitch,
    prepare_return_and_log_variables,
    send_discord_message,
)


if TYPE_CHECKING:
    import pandas as pd


def find_squad(  # noqa: PLR0914
    gameweek: int,
    budget: int = 1000,
    bench_weights: list[float] | None = None,
    weights_decays_base: list[float] | None = None,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], tuple[str, int]]:
    """

    Parameters
    ----------
    gameweek
        The gameweek under process.
    budget
        Total budget available for optimization.
    bench_weights
        Weights given to points of bench players.
    weights_decays_base
        Per GW decay for predicted points.

    Returns
    -------
        A tuple containing the lineup, bench and the captain.

    """
    if weights_decays_base is None:
        weights_decays_base = WEIGHTS_DECAYS_BASE
    if bench_weights is None:
        bench_weights = BENCH_WEIGHTS_ARRAY

    df_values: pd.DataFrame = prepare_df_for_optimization(
        gameweek, weights_decays_base
    )
    players, points, prices, positions, teams = prepare_common_lists_from_df(
        df_values
    )

    problem: pulp.LpProblem = pulp.LpProblem("squad_building", pulp.LpMaximize)
    lineup, bench_gk, bench_1, bench_2, bench_3, captain = (
        prepare_essential_lp_variables(players)
    )

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
        problem,
        lineup,
        bench_gk,
        bench_1,
        bench_2,
        bench_3,
        captain,
        positions,
        teams,
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
    ) = prepare_return_and_log_variables(
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
    this_season: Season = Seasons.SEASON_2425.value
    gw: int = 4
    eleven, subs, cap = find_squad(gw)
    logger.info("Starting Lineup: {}", eleven)
    logger.info("Bench: {}", subs)
    logger.info("Captain: {}", cap)
    eleven_players = build_fpl_lineup(eleven, this_season)
    sub_players = build_fpl_lineup(subs, this_season)
    pitch = prepare_pitch(eleven_players, sub_players, cap, this_season)
    message: str = "**Optimal Squad**"
    send_discord_message(message, pitch)
