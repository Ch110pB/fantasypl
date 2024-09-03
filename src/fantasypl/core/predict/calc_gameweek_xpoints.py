"""Functions to calculate expected points for players for each gameweek."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger
from scipy.stats import norm, poisson  # type: ignore[import-untyped]

from fantasypl.config.constants.folder_config import (
    DATA_FOLDER_FPL,
    DATA_FOLDER_REF,
    MODEL_FOLDER,
)
from fantasypl.config.constants.prediction_config import (
    FPL_POSITION_ID_DICT,
    POINTS_CS,
    POINTS_GOALS,
    POINTS_GOALS_CONCEDED,
    POINTS_SAVES,
)
from fantasypl.config.models.player import Player
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.save_helper import save_pandas


season: Season = Seasons.SEASON_2425.value


def calc_xpoints(gameweek: int) -> None:
    """

    Args:
    ----
        gameweek: Gameweek.

    """
    df_fpl_players: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FPL / season.folder / "players.csv"
    )
    with Path.open(DATA_FOLDER_REF / "players.json", "r") as fl:
        _list_players: list[Player] = [
            Player.model_validate(el) for el in json.load(fl).get("players")
        ]
    df_fpl_players["player"] = [
        {p.fpl_code: p.fbref_id for p in _list_players}.get(p)
        for p in df_fpl_players["code"]
    ]
    df_fpl_players["fpl_position"] = df_fpl_players["element_type"].map(
        FPL_POSITION_ID_DICT
    )
    df_fpl_players = df_fpl_players[
        [
            "player",
            "code",
            "photo",
            "fpl_position",
            "now_cost",
            "chance_of_playing_next_round",
            "selected_by_percent",
        ]
    ]
    df_fpl_players["chance_of_playing_next_round"] = df_fpl_players[
        "chance_of_playing_next_round"
    ].fillna(100.0)
    df_expected_stats: pd.DataFrame = pd.read_csv(
        MODEL_FOLDER
        / "predictions/player"
        / f"gameweek_{gameweek}/prediction_expected_stats.csv"
    )
    df_fpl_players = df_fpl_players.merge(
        df_expected_stats, on="player", how="left", validate="1:m"
    )

    df_fpl_players["prob_60"] = df_fpl_players["xmins"].apply(
        lambda x: 1 - norm.cdf(60, loc=x, scale=31.72) if x > 0 else 0
    )
    df_fpl_players["points_mins"] = df_fpl_players["prob_60"] * 2
    df_fpl_players["points_goals"] = df_fpl_players["xgoals"] * df_fpl_players[
        "fpl_position"
    ].map(POINTS_GOALS)
    df_fpl_players["points_assists"] = df_fpl_players["xgoals"] * 3
    df_fpl_players["points_yc"] = df_fpl_players["xyc"] * -1
    df_fpl_players["points_cs"] = (
        poisson.pmf(0, df_fpl_players["xgoals_vs"])
        * df_fpl_players["fpl_position"].map(POINTS_CS)
        * df_fpl_players["prob_60"]
    )
    df_fpl_players["points_goals_conceded"] = poisson.pmf(
        2, df_fpl_players["xgoals_vs"]
    ) * df_fpl_players["fpl_position"].map(POINTS_GOALS_CONCEDED)
    df_fpl_players["points_gk_saves"] = poisson.pmf(
        3, df_fpl_players["xsaves"]
    ) * df_fpl_players["fpl_position"].map(POINTS_SAVES)
    df_fpl_players["points"] = (
        df_fpl_players["points_mins"]
        + df_fpl_players["points_goals"]
        + df_fpl_players["points_assists"]
        + df_fpl_players["points_yc"]
        + df_fpl_players["points_cs"]
        + df_fpl_players["points_goals_conceded"]
        + df_fpl_players["points_gk_saves"]
    )
    df_fpl_players["points"] = (
        df_fpl_players["points"] * df_fpl_players["chance_of_playing_next_round"] / 100
    )
    save_pandas(
        df_fpl_players,
        MODEL_FOLDER
        / "predictions/player"
        / f"gameweek_{gameweek}"
        / "prediction_xpoints.csv",
    )
    logger.info("Expected points saved for all players.")


if __name__ == "__main__":
    calc_xpoints(4)
