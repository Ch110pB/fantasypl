"""Functions to find the optimal FPL transfers."""

import copy
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pulp  # type: ignore[import-untyped]
from loguru import logger
from scipy.stats import scoreatpercentile  # type: ignore[import-untyped]

from fantasypl.config.constants.folder_config import (
    DATA_FOLDER_FPL,
    DATA_FOLDER_REF,
    MODEL_FOLDER,
)
from fantasypl.config.constants.prediction_config import (
    BENCH_WEIGHTS_ARRAY,
    TRANSFER_GAIN_MINIMUM,
    TRANSFER_HIT_PENALTY_PERCENTILE,
    WEIGHTS_DECAYS_BASE,
)
from fantasypl.config.models.player import Player
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.prediction_helper import (
    add_count_constraints,
    add_other_constraints,
    arrange_return_and_log_variables,
    prepare_additional_lpvariables,
    prepare_common_lists_from_df,
    prepare_df_for_optimization,
    prepare_lp_variables,
)


if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


current_season: Season = Seasons.SEASON_2425.value

with Path.open(DATA_FOLDER_REF / "players.json", "r") as fl:
    _list_players: list[Player] = [
        Player.model_validate(el) for el in json.load(fl).get("players")
    ]


def calculate_sell_price(current_price: int, buy_price: int) -> int:
    """

    Args:
    ----
        current_price: Current price of a player.
        buy_price: Buying price of the player.

    Returns:
    -------
        Selling price of the player.

    """
    if current_price <= buy_price:
        return current_price
    return buy_price + (current_price - buy_price) // 2


def prepare_data_for_current_team(
    last_gameweek: int,
) -> tuple[dict[int, int], dict[int, int], list[int], int, int]:
    """

    Args:
    ----
        last_gameweek: Gameweek.

    Returns:
    -------
        Relevant current team details for optimization.

    """
    df_fpl: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FPL / current_season.folder / "players.csv"
    )
    df_fpl = df_fpl[["id", "cost_change_start", "now_cost"]]
    df_fpl["buy_price"] = df_fpl["now_cost"] - df_fpl["cost_change_start"]
    buy_prices: dict[int, int] = (
        df_fpl[["id", "buy_price"]].set_index("id").to_dict()["buy_price"]
    )
    current_prices: dict[int, int] = (
        df_fpl[["id", "now_cost"]].set_index("id").to_dict()["now_cost"]
    )

    with Path.open(
        MODEL_FOLDER
        / "predictions/player"
        / f"gameweek_{last_gameweek}"
        / "team_last_gw.json",
        "r",
    ) as f:
        current_team_data: dict[str, Any] = json.load(f)
    current_team: list[int] = [el["element"] for el in current_team_data["picks"]]
    current_team_buy_prices: dict[int, int] = {
        el["element"]: buy_prices[el["element"]] for el in current_team_data["picks"]
    }

    with Path.open(
        MODEL_FOLDER
        / "predictions/player"
        / f"gameweek_{last_gameweek}"
        / "team_transfers.json",
        "r",
    ) as f:
        transfers_data: list[dict[str, Any]] = json.load(f)
    transfers_buy_prices: dict[int, int] = {
        el["element_in"]: el["element_in_cost"] for el in transfers_data
    }
    current_team_buy_prices.update(transfers_buy_prices)
    current_team_sell_prices: dict[int, int] = {
        k: calculate_sell_price(current_prices[k], current_team_buy_prices[k])
        for k in current_team_buy_prices
    }

    now_buy_prices: dict[int, int] = copy.deepcopy(current_prices)
    sell_prices: dict[int, int] = copy.deepcopy(current_prices)
    now_buy_prices.update(current_team_buy_prices)
    sell_prices.update(current_team_sell_prices)

    transfers_count_dict: dict[int, int] = dict(
        Counter([t["event"] for t in transfers_data])
    )
    free_transfers: int = 1
    for gw in range(1, last_gameweek + 1):
        free_transfers += 1 - transfers_count_dict.get(gw, 0)

    return (
        now_buy_prices,
        sell_prices,
        current_team,
        current_team_data["entry_history"]["bank"],
        free_transfers,
    )


def find_optimal_transfers(  # noqa: PLR0914
    gameweek: int,
    bench_weights: list[float] | None = None,
    weights_decays_base: list[float] | None = None,
    transfer_penalty_percentile: float | None = None,
    transfer_gain_minimum: float | None = None,
) -> tuple[list[str], list[str], str, list[str], list[str], list[str]]:
    """

    Args:
    ----
        gameweek: Gameweek.
        bench_weights: Weights given to points of bench players.
        weights_decays_base: Per GW decay for predicted points.
        transfer_penalty_percentile:
                Transfer penalty on additional hits as percentile of predictions.
        transfer_gain_minimum: Minimum points gain to warrant a transfer.


    Returns:
    -------
        A tuple containing the lineup, bench, the captain and the transfers.

    """
    df_fpl: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FPL / current_season.folder / "players.csv"
    )
    _code_to_id_dict: dict[int, int] = (
        df_fpl[["id", "code"]].set_index("code").to_dict()["id"]
    )
    _id_to_code_dict: dict[int, int] = (
        df_fpl[["id", "code"]].set_index("id").to_dict()["code"]
    )
    if weights_decays_base is None:
        weights_decays_base = WEIGHTS_DECAYS_BASE
    if bench_weights is None:
        bench_weights = BENCH_WEIGHTS_ARRAY
    if transfer_penalty_percentile is None:
        transfer_penalty_percentile = TRANSFER_HIT_PENALTY_PERCENTILE
    if transfer_gain_minimum is None:
        transfer_gain_minimum = TRANSFER_GAIN_MINIMUM

    df_values: pd.DataFrame = prepare_df_for_optimization(gameweek, weights_decays_base)
    df_values["id"] = df_values["code"].map(_code_to_id_dict)

    dict_buy_prices, dict_sell_prices, _current_team, itb, free_transfers = (
        prepare_data_for_current_team(gameweek - 1)
    )
    df_values["buy_price"] = df_values["id"].map(dict_buy_prices)
    df_values["sell_price"] = df_values["id"].map(dict_sell_prices)

    players, points, prices, positions, teams = prepare_common_lists_from_df(df_values)
    buy_prices: npt.NDArray[np.int32] = df_values["buy_price"].to_numpy()
    sell_prices: npt.NDArray[np.int32] = df_values["sell_price"].to_numpy()

    problem: pulp.LpProblem = pulp.LpProblem("transfers", pulp.LpMaximize)

    lineup, bench_gk, bench_1, bench_2, bench_3, captain = prepare_lp_variables(players)
    initial_squad, transfers_out, transfers_in_free, transfers_in_hit = (
        prepare_additional_lpvariables(players)
    )

    problem.setObjective(
        points @ (lineup + captain)
        + (bench_weights[0] * points) @ bench_gk
        + (bench_weights[1] * points) @ bench_1
        + (bench_weights[2] * points) @ bench_2
        + (bench_weights[3] * points) @ bench_3
        - scoreatpercentile(points, transfer_penalty_percentile) * sum(transfers_in_hit)
    )
    problem = add_count_constraints(
        problem, lineup, bench_gk, bench_1, bench_2, bench_3, captain
    )
    problem = add_other_constraints(
        problem, lineup, bench_gk, bench_1, bench_2, bench_3, captain, positions, teams
    )

    current_team: set[int] = {_id_to_code_dict[el] for el in _current_team}
    for i in range(len(players)):
        problem.addConstraint(initial_squad[i] == int(players[i] in current_team))

    problem.addConstraint(sum(transfers_in_free) <= free_transfers)
    problem.addConstraint(
        sell_prices @ transfers_out + itb
        >= buy_prices @ (transfers_in_free + transfers_in_hit)
    )
    problem.addConstraint(
        points @ (lineup + bench_gk + bench_1 + bench_2 + bench_3)
        - points @ initial_squad
        >= transfer_gain_minimum
    )

    transfer_out_in_squad_expressions: npt.NDArray[pulp.LpVariable] = (
        initial_squad - transfers_out
    )
    for expr in transfer_out_in_squad_expressions:
        problem.addConstraint(expr >= 0)
    transfer_in_not_in_squad_expressions: npt.NDArray[pulp.LpVariable] = (
        initial_squad + transfers_in_free + transfers_in_hit
    )
    for expr in transfer_in_not_in_squad_expressions:
        problem.addConstraint(expr <= 1)
    transfer_zero_sum_expressions: npt.NDArray[pulp.LpVariable] = (
        lineup
        + bench_gk
        + bench_1
        + bench_2
        + bench_3
        + transfers_out
        - initial_squad
        - transfers_in_hit
        - transfers_in_free
    )
    for expr in transfer_zero_sum_expressions:
        problem.addConstraint(expr == 0)

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
    selected_players: list[str] = [
        v.name for v in problem.variables() if v.varValue == 1
    ]
    transfers_out_players: list[str] = [
        el.fpl_web_name
        for el in _list_players
        if f"out{el.fpl_code}" in selected_players
    ]
    transfers_in_free_players: list[str] = [
        el.fpl_web_name
        for el in _list_players
        if f"ft{el.fpl_code}" in selected_players
    ]
    transfers_in_hit_players: list[str] = [
        el.fpl_web_name
        for el in _list_players
        if f"hit{el.fpl_code}" in selected_players
    ]

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
    return (
        lineup_players,
        bench_players,
        captain_player,
        transfers_out_players,
        transfers_in_free_players,
        transfers_in_hit_players,
    )


if __name__ == "__main__":
    eleven, subs, cap, out, ft, hit = find_optimal_transfers(4)
    logger.info("Starting Lineup: {}", eleven)
    logger.info("Bench: {}", subs)
    logger.info("Captain: {}", cap)
    logger.info("Out: {}", out)
    logger.info("In on FT: {}", ft)
    logger.info("In on Hit: {}", hit)
