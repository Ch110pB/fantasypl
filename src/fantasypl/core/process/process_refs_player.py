"""Functions for creating Player objects JSON."""

import json
from pathlib import Path

import pandas as pd
import rich.progress
from loguru import logger

from fantasypl.config.constants.folder_config import (
    DATA_FOLDER_FBREF,
    DATA_FOLDER_FPL,
    DATA_FOLDER_REF,
)
from fantasypl.config.models.player import Player
from fantasypl.config.models.season import Season, Seasons
from fantasypl.config.references.player_refs import FBREF_FPL_PLAYER_REF_DICT
from fantasypl.utils.save_helper import save_json


def get_player_references(season: Season) -> None:
    """

    Args:
    ----
        season: Season.

    """
    players: list[Player] = []
    dfs: list[pd.DataFrame] = []
    for s in [Seasons.SEASON_2324.value, Seasons.SEASON_2425.value]:
        df_fpl_players: pd.DataFrame = pd.read_csv(
            DATA_FOLDER_FPL / s.folder / "players.csv",
        )
        df_fpl_players = df_fpl_players.rename(
            columns={
                "code": "fpl_code",
                "full_name": "fpl_full_name",
                "web_name": "fpl_web_name",
            },
        )[["fpl_code", "fpl_full_name", "fpl_web_name"]]
        dfs.append(df_fpl_players)
    df_fpl: pd.DataFrame = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if df_fpl.empty:
        logger.error("FPL player dataframes not found!!!")
        return
    df_fpl["fbref_id"] = df_fpl["fpl_code"].map(
        {v: k for k, v in FBREF_FPL_PLAYER_REF_DICT.items()},
    )
    player_ids: dict[str, str] = dict.fromkeys(df_fpl["fbref_id"].dropna().tolist(), "")
    for k in rich.progress.track(player_ids, "Reading FBRef player JSONs"):
        with Path.open(
            DATA_FOLDER_FBREF / season.folder / "player_season" / f"{k}.json", "r"
        ) as f:
            name_: str = json.load(f).get("name")
            player_ids[k] = name_
    df_fpl["fbref_name"] = df_fpl["fbref_id"].map(player_ids)
    players += [
        Player.model_validate(row)
        for row in df_fpl.dropna(how="any").to_dict(orient="records")
    ]
    players_str: list[dict[str, Player]] = [
        player.model_dump() for player in list(set(players))
    ]
    save_json({"players": players_str}, DATA_FOLDER_REF / "players.json")
    logger.info("Players class object references JSON successfully saved")


if __name__ == "__main__":
    get_player_references(Seasons.SEASON_2324.value)
