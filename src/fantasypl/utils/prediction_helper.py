"""Helper functions for prediction and optimization."""

import json
import operator
from functools import reduce
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
from loguru import logger
from PIL import Image
from pulp import (  # type: ignore[import-untyped]
    LpBinary,
    LpProblem,
    LpVariable,
    value,
)

from fantasypl.config.constants import (
    DATA_FOLDER_FPL,
    FPL_POSITION_ID_DICT,
    MAX_DEF_COUNT,
    MAX_FWD_COUNT,
    MAX_GKP_COUNT,
    MAX_MID_COUNT,
    MAX_SAME_CLUB_COUNT,
    MIN_DEF_COUNT,
    MIN_FWD_COUNT,
    MIN_GKP_COUNT,
    MIN_MID_COUNT,
    MODEL_FOLDER,
    TOTAL_DEF_COUNT,
    TOTAL_FWD_COUNT,
    TOTAL_GKP_COUNT,
    TOTAL_LINEUP_COUNT,
    TOTAL_MID_COUNT,
)
from fantasypl.config.constants.folder_config import ROOT_FOLDER
from fantasypl.config.schemas import Player, Season, Team
from fantasypl.utils.modeling_helper import get_list_players, get_list_teams


_list_teams: list[Team] = get_list_teams()
_list_players: list[Player] = get_list_players()


def pad_lists(
    row: pd.Series,  # type: ignore[type-arg]
    df_prev_agg: pd.DataFrame,
    col: str,
    group_col: str,
) -> Any:  # noqa: ANN401
    """
    Pad the list of features with aggregated stats from last season.

    Parameters
    ----------
    row
        A row of the dataframe for the current season.
    df_prev_agg
        The aggregate dataframe from the previous season.
    col
        The column name containing the list.
    group_col
        The column to group by.

    Returns
    -------
        The row with the list in the corresponding column
        with values padded from previous season.

    """
    prev_season_value: str | int | float = df_prev_agg.at[row[group_col], col]  # noqa: PD008
    if len(row[col]) < 5:  # noqa: PLR2004
        row[col] = [prev_season_value] * (5 - len(row[col])) + row[col]
    return row[col]


def prepare_df_for_optimization(
    gameweek: int, weights_decays_base: list[float]
) -> pd.DataFrame:
    """
    Prepare dataframe for optimization.

    Parameters
    ----------
    gameweek
        The gameweek under process.
    weights_decays_base
        Per GW decay array for predicted points.

    Returns
    -------
        A dataframe containing the features for the optimization problem.

    """
    df_expected_points: pd.DataFrame = pd.read_csv(
        MODEL_FOLDER
        / "predictions/player"
        / f"gameweek_{gameweek}"
        / "prediction_xpoints.csv",
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
    Prepare lists for optimization.

    Parameters
    ----------
    df_values
        The prepared dataframe for optimization.

    Returns
    -------
        A tuple containing players, points, prices, positions and teams

    """
    players: npt.NDArray[np.int32] = df_values["code"].to_numpy()
    points: npt.NDArray[np.float32] = df_values["weighted_points"].to_numpy()
    prices: npt.NDArray[np.int32] = df_values["now_cost"].to_numpy()
    positions: npt.NDArray[np.str_] = df_values["fpl_position"].to_numpy()
    teams: npt.NDArray[np.str_] = df_values["team"].to_numpy()
    return players, points, prices, positions, teams


def helper_create_lp_variables(
    prefixes: list[str],
    players: npt.NDArray[np.int32],
) -> list[npt.NDArray[LpVariable]]:
    """
    Create LP variables helper function.

    Parameters
    ----------
    prefixes
        List of prefixes.
    players
        The list of players.

    Returns
    -------
        List of LP variables.

    """
    return [
        np.array([LpVariable(f"{prefix}{pl}", cat=LpBinary) for pl in players])
        for prefix in prefixes
    ]


def prepare_essential_lp_variables(
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
    Create essential LP variables.

    Parameters
    ----------
    players
        The list of players.

    Returns
    -------
        A tuple containing the LP variables for lineup and bench.

    """
    prefixes: list[str] = ["l", "bg", "bf", "bs", "bt", "c"]
    lineup, bench_gk, bench_1, bench_2, bench_3, captain = (
        helper_create_lp_variables(prefixes, players)
    )
    return lineup, bench_gk, bench_1, bench_2, bench_3, captain


def prepare_additional_lp_variables(
    players: npt.NDArray[np.int32],
) -> tuple[
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
    npt.NDArray[LpVariable],
]:
    """
    Create additional LP variables.

    Parameters
    ----------
    players
        The list of players.

    Returns
    -------
        A tuple containing the LP variables for transfers.

    """
    prefixes = ["sq", "out", "ft", "hit"]
    initial_squad, transfers_out, transfers_in_free, transfers_in_hit = (
        helper_create_lp_variables(prefixes, players)
    )
    return initial_squad, transfers_out, transfers_in_free, transfers_in_hit


def add_count_constraints(  # noqa: PLR0913, PLR0917
    problem: LpProblem,
    lineup: npt.NDArray[LpVariable],
    bench_gk: npt.NDArray[LpVariable],
    bench_1: npt.NDArray[LpVariable],
    bench_2: npt.NDArray[LpVariable],
    bench_3: npt.NDArray[LpVariable],
    captain: npt.NDArray[LpVariable],
) -> LpProblem:
    """
    Create LP constraints for counts.

    Parameters
    ----------
    problem
        The LP problem.
    lineup
        The LP variable for lineup.
    bench_gk
        The LP variable for the bench gk.
    bench_1
        The LP variable for the first bench.
    bench_2
        The LP variable for the second bench.
    bench_3
        The LP variable for the third bench.
    captain
        The LP variable for the captain.

    Returns
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


def helper_add_positional_constraints(  # noqa: PLR0913, PLR0917
    problem: LpProblem,
    mask: npt.NDArray[np.bool],
    lineup: npt.NDArray[LpVariable],
    bench: list[npt.NDArray[LpVariable]],
    min_count: int,
    max_count: int,
    total_count: int,
) -> LpProblem:
    """
    Create LP constraints for positions.

    Parameters
    ----------
    problem
        The LP problem.
    mask
        Position mask.
    lineup
        The LP variable for lineup.
    bench
        The LP variables array for bench.
    min_count
        Minimum count for position.
    max_count
        Maximum count for position.
    total_count
        Total count for position.

    Returns
    -------
        The LP problem with positional constraints.

    """
    problem.addConstraint(mask @ lineup >= min_count)
    problem.addConstraint(mask @ lineup <= max_count)
    problem.addConstraint(
        mask @ (lineup + reduce(operator.add, bench)) == total_count,
    )
    return problem


def add_other_constraints(  # noqa: PLR0913, PLR0917
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
    Create other LP constraints.

    Parameters
    ----------
    problem
        The LP problem.
    lineup
        The LP variable for lineup.
    bench_gk
        The LP variable for the bench gk.
    bench_1
        The LP variable for the first bench.
    bench_2
        The LP variable for the second bench.
    bench_3
        The LP variable for the third bench.
    captain
        The LP variable for the captain.
    positions
        The list of positions.
    teams
        The list of teams.

    Returns
    -------
        The LP problem with membership, positional
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

    problem = helper_add_positional_constraints(
        problem,
        np.array(positions == "GKP"),
        lineup,
        [bench_gk],
        MIN_GKP_COUNT,
        MAX_GKP_COUNT,
        TOTAL_GKP_COUNT,
    )
    problem = helper_add_positional_constraints(
        problem,
        np.array(positions == "DEF"),
        lineup,
        [bench_1, bench_2, bench_3],
        MIN_DEF_COUNT,
        MAX_DEF_COUNT,
        TOTAL_DEF_COUNT,
    )
    problem = helper_add_positional_constraints(
        problem,
        np.array(positions == "MID"),
        lineup,
        [bench_1, bench_2, bench_3],
        MIN_MID_COUNT,
        MAX_MID_COUNT,
        TOTAL_MID_COUNT,
    )
    problem = helper_add_positional_constraints(
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
            <= MAX_SAME_CLUB_COUNT,
        )

    return problem


def prepare_return_and_log_variables(  # noqa: PLR0913, PLR0917
    problem: LpProblem,
    lineup: npt.NDArray[LpVariable],
    bench_gk: npt.NDArray[LpVariable],
    bench_1: npt.NDArray[LpVariable],
    bench_2: npt.NDArray[LpVariable],
    bench_3: npt.NDArray[LpVariable],
) -> tuple[
    npt.NDArray[np.float32],
    list[tuple[str, int]],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    list[tuple[str, int]],
    tuple[str, int],
]:
    """
    Prepare variables to return or log.

    Parameters
    ----------
    problem
        The LP problem.
    lineup
        The LP variable for lineup.
    bench_gk
        The LP variable for the bench gk.
    bench_1
        The LP variable for the first bench.
    bench_2
        The LP variable for the second bench.
    bench_3
        The LP variable for the third bench.

    Returns
    -------
        A tuple containing all variables required for return
        and logging purposes.

    """
    optimal_lineup: npt.NDArray[np.float32] = np.array([
        value(var) for var in lineup
    ])
    optimal_bench_gk: npt.NDArray[np.float32] = np.array([
        value(var) for var in bench_gk
    ])
    optimal_bench_1: npt.NDArray[np.float32] = np.array([
        value(var) for var in bench_1
    ])
    optimal_bench_2: npt.NDArray[np.float32] = np.array([
        value(var) for var in bench_2
    ])
    optimal_bench_3: npt.NDArray[np.float32] = np.array([
        value(var) for var in bench_3
    ])
    selected_players: list[str] = [
        v.name for v in problem.variables() if v.varValue == 1
    ]
    lineup_players: list[tuple[str, int]] = [
        (el.fpl_web_name, el.fpl_code)
        for el in _list_players
        if f"l{el.fpl_code}" in selected_players
    ]
    bench_players: list[tuple[str, int]] = (
        [
            (el.fpl_web_name, el.fpl_code)
            for el in _list_players
            if f"bg{el.fpl_code}" in selected_players
        ]
        + [
            (el.fpl_web_name, el.fpl_code)
            for el in _list_players
            if f"bf{el.fpl_code}" in selected_players
        ]
        + [
            (el.fpl_web_name, el.fpl_code)
            for el in _list_players
            if f"bs{el.fpl_code}" in selected_players
        ]
        + [
            (el.fpl_web_name, el.fpl_code)
            for el in _list_players
            if f"bt{el.fpl_code}" in selected_players
        ]
    )
    captain_player: tuple[str, int] = next(
        (el.fpl_web_name, el.fpl_code)
        for el in _list_players
        if f"c{el.fpl_code}" in selected_players
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


def add_team_and_position_to_player(
    player: tuple[str, int],
    season: Season,
) -> tuple[str, int, int, str]:
    """
    Add team ID and position to selected players.

    Parameters
    ----------
    player
        Tuple containing player FPL web name and FPL code.
    season
        The season under process.

    Returns
    -------
        A Tuple containing player FPL web name, team FPL code
        and FPL position.

    """
    player_fpl_code: int = player[1]
    df_fpl_players: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FPL / season.folder / "players.csv"
    )
    player_team_code: int = (
        df_fpl_players[df_fpl_players["code"] == player_fpl_code]["team_code"]
        .to_numpy()
        .item(0)
    )
    player_position_code: int = (
        df_fpl_players[df_fpl_players["code"] == player_fpl_code][
            "element_type"
        ]
        .to_numpy()
        .item(0)
    )
    return (
        player[0],
        player_fpl_code,
        player_team_code,
        FPL_POSITION_ID_DICT[player_position_code],
    )


def build_fpl_lineup(
    players: list[tuple[str, int]],
    season: Season,
) -> tuple[
    list[tuple[str, int, int]],
    list[tuple[str, int, int]],
    list[tuple[str, int, int]],
    list[tuple[str, int, int]],
]:
    """
    Build the final FPL lineup.

    Parameters
    ----------
    players
         List of tuples containing player FPL web name, team
         FPL code and FPL position.
    season
        The season under process.

    Returns
    -------
        Lists of players per FPL position.

    """
    fpl_players: list[tuple[str, int, int, str]] = [
        add_team_and_position_to_player(pl, season) for pl in players
    ]
    fpl_gk: list[tuple[str, int, int]] = [
        (el[0], el[1], el[2]) for el in fpl_players if el[3] == "GKP"
    ]
    fpl_df: list[tuple[str, int, int]] = [
        (el[0], el[1], el[2]) for el in fpl_players if el[3] == "DEF"
    ]
    fpl_mf: list[tuple[str, int, int]] = [
        (el[0], el[1], el[2]) for el in fpl_players if el[3] == "MID"
    ]
    fpl_fw: list[tuple[str, int, int]] = [
        (el[0], el[1], el[2]) for el in fpl_players if el[3] == "FWD"
    ]
    return fpl_gk, fpl_df, fpl_mf, fpl_fw


def send_discord_message(text: str, images: list[Image.Image]) -> None:
    """
    Send a discord message.

    Parameters
    ----------
    text
        The message to send to Discord.
    images
        The images to send to Discord.

    """
    with Path.open(ROOT_FOLDER / "discord_authorization.json", "r") as f:
        auth_dict: dict[str, str] = json.load(f)
    url: str = f"https://discord.com/api/v10/channels/{auth_dict["channel_id"]}/messages"
    data: dict[str, str] = {"content": text}
    files = {}
    for i, image in enumerate(images, start=1):
        image_file_bytes: BytesIO = BytesIO()
        image.save(image_file_bytes, format="PNG")
        files[f"file{i}"] = (f"image{i}.png", image_file_bytes.getvalue())
    headers: dict[str, str] = {"authorization": auth_dict["token"]}

    response: requests.Response = requests.post(
        url,
        data=data,
        files=files,
        headers=headers,
        timeout=5,
    )
    logger.info("Discord message status code: {}", response.status_code)
