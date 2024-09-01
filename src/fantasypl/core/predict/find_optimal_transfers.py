"""Functions to find the optimal FPL transfers."""

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
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
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.prediction_helper import (
    add_position_constraints,
    prepare_df_for_optimization,
)


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


# noinspection DuplicatedCode
def find_optimal_transfers(  # noqa: PLR0914, PLR0915
    gameweek: int,
    bench_weights: list[float] | None = None,
    transfer_penalty_percentile: int | None = None,
) -> tuple[list[str], list[str], str, list[str], list[str], list[str]]:
    """

    Args:
    ----
        gameweek: Gameweek.
        bench_weights: Weights given to points of bench players.
        transfer_penalty_percentile:
                Transfer penalty on additional hits
                (-4 is the value in official FPL).

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

    df_values: pd.DataFrame = prepare_df_for_optimization(gameweek)
    df_values["id"] = df_values["code"].map(_code_to_id_dict)

    dict_buy_prices, dict_sell_prices, _current_team, itb, free_transfers = (
        prepare_data_for_current_team(gameweek - 1)
    )
    df_values["buy_price"] = df_values["id"].map(dict_buy_prices)
    df_values["sell_price"] = df_values["id"].map(dict_sell_prices)

    players: npt.NDArray[np.int32] = df_values["code"].to_numpy()
    points: npt.NDArray[np.float32] = df_values["weighted_points"].to_numpy()
    prices: npt.NDArray[np.int32] = df_values["now_cost"].to_numpy()
    positions: npt.NDArray[np.str_] = df_values["fpl_position"].to_numpy()
    teams: npt.NDArray[np.str_] = df_values["team"].to_numpy()
    buy_prices: npt.NDArray[np.int32] = df_values["buy_price"].to_numpy()
    sell_prices: npt.NDArray[np.int32] = df_values["sell_price"].to_numpy()

    problem: pulp.LpProblem = pulp.LpProblem("transfers", pulp.LpMaximize)

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
    initial_squad: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"sq{pl}", cat=pulp.LpBinary) for pl in players
    ])
    transfers_out: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"out{pl}", cat=pulp.LpBinary) for pl in players
    ])
    transfers_in_free: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"ft{pl}", cat=pulp.LpBinary) for pl in players
    ])
    transfers_in_hit: npt.NDArray[pulp.LpVariable] = np.array([
        pulp.LpVariable(f"hit{pl}", cat=pulp.LpBinary) for pl in players
    ])

    if bench_weights is None:
        bench_weights = [0.03, 0.21, 0.1, 0.002]
    if transfer_penalty_percentile is None:
        transfer_penalty_percentile = 77

    problem.setObjective(
        points @ (lineup + captain)
        + (bench_weights[0] * points) @ bench_gk
        + (bench_weights[1] * points) @ bench_1
        + (bench_weights[2] * points) @ bench_2
        + (bench_weights[3] * points) @ bench_3
        - scoreatpercentile(points, transfer_penalty_percentile) * sum(transfers_in_hit)
    )
    problem.addConstraint(sum(lineup) == TOTAL_LINEUP_COUNT)
    problem.addConstraint(sum(bench_gk) == 1)
    problem.addConstraint(sum(bench_1) == 1)
    problem.addConstraint(sum(bench_2) == 1)
    problem.addConstraint(sum(bench_3) == 1)
    problem.addConstraint(sum(captain) == 1)

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

    current_team: set[int] = {_id_to_code_dict[el] for el in _current_team}
    for i in range(len(players)):
        problem.addConstraint(initial_squad[i] == int(players[i] in current_team))

    problem.addConstraint(sum(transfers_in_free) <= free_transfers)
    problem.addConstraint(
        sell_prices @ transfers_out + itb
        >= buy_prices @ (transfers_in_free + transfers_in_hit)
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
