"""Functions to calculate player stats averages from last season."""

import json
from functools import reduce
from pathlib import Path

import pandas as pd
import rich.progress

from fantasypl.config.constants.folder_config import (
    DATA_FOLDER_FBREF,
    DATA_FOLDER_FPL,
    DATA_FOLDER_REF,
)
from fantasypl.config.constants.mapping_config import FBREF_LEAGUE_STRENGTH_DICT
from fantasypl.config.models.player import Player
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.save_helper import save_pandas


with Path.open(DATA_FOLDER_REF / "players.json", "r") as f:
    _list_players: list[Player] = [
        Player.model_validate(el) for el in json.load(f).get("players")
    ]

current_season: Season = Seasons.SEASON_2425.value


def process_stat(  # noqa: PLR0917
    season: Season,
    player_id: str,
    stat: str,
    rename_dict: dict[str, str],
    cols: list[str],
    game_count_col: str = "minutes_90s",
) -> pd.DataFrame:
    """

    Args:
    ----
        season: Season.
        player_id: Player FBRef ID.
        stat: File name to look at.
        rename_dict: Column rename dictionary.
        cols: Columns to be selected.
        game_count_col: Total number of games played.

    Returns:
    -------
        A dataframe with per90 columns for a stat.

    """
    try:
        df_stats: pd.DataFrame = pd.read_csv(
            DATA_FOLDER_FBREF
            / season.folder
            / "player_season"
            / f"{player_id}_{stat}.csv"
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame([{"player": player_id, **dict.fromkeys(cols, 0)}])
    except FileNotFoundError:
        return pd.DataFrame([{"player": player_id, **dict.fromkeys(cols, 0)}])
    df_stats = df_stats.dropna(subset=["year_id"])
    df_stats = df_stats[
        (df_stats["year_id"].isin(["2023-2024", "2023"]))
        & (~df_stats["comp_level"].str.contains("Jr."))
    ]
    if df_stats.empty:
        return pd.DataFrame([{"player": player_id, **dict.fromkeys(cols, 0)}])
    df_stats = df_stats.rename(columns=rename_dict)
    for col in cols:
        if col not in df_stats.columns:
            df_stats[col] = 0
    df_stats = df_stats[["country", "comp_level", game_count_col, *cols]]
    df_stats["player"] = player_id
    for col in cols:
        df_stats[col] = df_stats[col].fillna(0)
        if col not in {"starts", "minutes"}:
            df_stats[col] = df_stats.apply(
                lambda row, c=col: row[c] / row[game_count_col]
                if row[game_count_col] > 0
                else 0.0,
                axis=1,
            )
        df_stats[col] = df_stats.apply(
            lambda row, c=col: row[c]
            * FBREF_LEAGUE_STRENGTH_DICT[row["country"] + "_" + row["comp_level"]]
            / FBREF_LEAGUE_STRENGTH_DICT["eng ENG_1. Premier League"],
            axis=1,
        )
        df_stats[col] = df_stats[col].mean()
    return df_stats.drop(columns=["country", "comp_level", game_count_col]).head(1)


def build_stats(season: Season) -> None:  # noqa: PLR0914
    """

    Args:
    ----
        season: Season.

    """
    df_fpl_players: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FPL / current_season.folder / "players.csv"
    )[["code"]]
    df_fpl_players["player"] = [
        next((el.fbref_id for el in _list_players if el.fpl_code == x), None)
        for x in df_fpl_players["code"]
    ]
    df_fpl_players = df_fpl_players.drop(columns=["code"])

    dfs: list[pd.DataFrame] = []
    for player in rich.progress.track(
        df_fpl_players["player"].unique().tolist(),
        "Processing player previous season: ",
    ):
        try:
            with Path.open(
                DATA_FOLDER_FBREF / season.folder / f"player_season/{player}.json", "r"
            ) as fl:
                dict_: dict[str, str] = json.load(fl)
            df_: pd.DataFrame = pd.DataFrame([
                {"player": player, "position": str(dict_["position"])[:2]}
            ])
        except FileNotFoundError:
            continue

        df_playing_time: pd.DataFrame = process_stat(
            season,
            player,
            "playing_time",
            {
                "header_playing_minutes_per_game": "minutes",
                "header_starts_games_starts": "starts",
            },
            ["minutes", "starts"],
            "header_playing_minutes_90s",
        )

        df_standard: pd.DataFrame = process_stat(
            season,
            player,
            "standard",
            {
                "header_progression_progressive_carries": "progressive_carries",
                "header_progression_progressive_passes": "progressive_passes",
            },
            ["progressive_carries", "progressive_passes"],
            "header_playing_minutes_90s",
        )

        df_shooting: pd.DataFrame = process_stat(
            season,
            player,
            "shooting",
            {
                "header_standard_shots_on_target": "shots_on_target",
                "header_expected_npxg": "npxg",
                "header_standard_pens_att": "pens_taken",
                "header_standard_pens_made": "pens_scored",
            },
            ["shots_on_target", "npxg", "pens_taken", "pens_scored"],
        )

        df_passing: pd.DataFrame = process_stat(
            season,
            player,
            "passing",
            {"header_expected_pass_xa": "pass_xa", "assisted_shots": "key_passes"},
            ["pass_xa", "key_passes"],
        )

        df_gca: pd.DataFrame = process_stat(
            season,
            player,
            "gca",
            {"header_sca_sca": "sca", "header_gca_gca": "gca"},
            ["sca", "gca"],
        )

        df_misc: pd.DataFrame = process_stat(
            season,
            player,
            "misc",
            {
                "header_performance_cards_yellow": "yellow_cards",
                "header_performance_cards_red": "red_cards",
                "header_performance_fouls": "fouls",
            },
            ["yellow_cards", "red_cards", "fouls"],
        )

        df_defense: pd.DataFrame = process_stat(
            season,
            player,
            "defense",
            {
                "header_tackles_tackles_won": "tackles_won",
                "header_blocks_blocks": "blocks",
            },
            ["tackles_won", "blocks", "interceptions", "clearances"],
        )

        df_keeper: pd.DataFrame = process_stat(
            season,
            player,
            "keeper",
            {
                "header_performance_gk_saves": "gk_saves",
            },
            ["gk_saves"],
            "header_playing_minutes_90s",
        )

        df_keeperadv: pd.DataFrame = process_stat(
            season,
            player,
            "keeper_adv",
            {
                "header_expected_gk_psxg": "gk_psxg",
            },
            ["gk_psxg"],
        )

        df_player: pd.DataFrame = reduce(
            lambda left, right: left.merge(
                right,
                on=["player"],
                how="left",
                validate="1:1",
            ),
            [
                df_,
                df_standard,
                df_playing_time,
                df_shooting,
                df_passing,
                df_gca,
                df_misc,
                df_defense,
                df_keeper,
                df_keeperadv,
            ],
        )
        dfs.append(df_player)
    df_all_stats: pd.DataFrame = (
        pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    )
    df_fpl_players = df_fpl_players.merge(
        df_all_stats, on="player", how="left", validate="1:1"
    )
    fpath: Path = DATA_FOLDER_FBREF / season.folder / "player_seasonal_stats.csv"
    save_pandas(df_fpl_players, fpath)


if __name__ == "__main__":
    build_stats(Seasons.SEASON_2324.value)
