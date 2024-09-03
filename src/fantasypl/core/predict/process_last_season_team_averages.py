"""Functions to calculate team stats averages from last season."""

from functools import reduce
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF
from fantasypl.config.constants.mapping_config import FBREF_LEAGUE_OPTA_STRENGTH_DICT
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.save_helper import save_pandas


if TYPE_CHECKING:
    from pathlib import Path


def process_stat(  # noqa: PLR0917
    season: Season,
    league_id: int,
    stat: str,
    rename_dict: dict[str, str],
    cols: list[str],
    game_count_col: str = "minutes_90s",
) -> pd.DataFrame:
    """

    Args:
    ----
        season: Season.
        league_id: FBRef league ID. (9=PL, 10=Championship)
        stat: File name to look at.
        rename_dict: Column rename dictionary.
        cols: Columns to be selected.
        game_count_col: Total number of games played.

    Returns:
    -------
        A dataframe with per90 columns for a stat.

    """
    df_: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FBREF
        / season.folder
        / "team_season"
        / str(league_id)
        / f"{stat}.csv"
    )
    if stat.endswith("against"):
        df_["team"] = df_["team"].str.replace("vs ", "")
    df_ = df_.rename(columns=rename_dict)
    df_ = df_[["team", game_count_col, *cols]]
    for col in cols:
        df_[col] /= df_[game_count_col]
        if league_id == 10:  # noqa: PLR2004
            df_[col] *= (
                (FBREF_LEAGUE_OPTA_STRENGTH_DICT["eng ENG_2. Championship"]) ** 2
                / (FBREF_LEAGUE_OPTA_STRENGTH_DICT["eng ENG_1. Premier League"]) ** 2
            )
    return df_.drop(columns=game_count_col)


def build_team_features_prediction(season: Season) -> None:
    """

    Args:
    ----
        season: Season.

    """
    df_formation: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FBREF / season.folder / "team_season" / "formations.csv"
    )

    dfs_leagues: list[pd.DataFrame] = []
    for league_id in [9, 10]:
        df_standard: pd.DataFrame = pd.read_csv(
            DATA_FOLDER_FBREF
            / season.folder
            / "team_season"
            / str(league_id)
            / "standard.csv"
        )
        df_standard = df_standard[["team", "possession"]]
        if league_id == 10:  # noqa: PLR2004
            df_standard["possession"] *= (
                FBREF_LEAGUE_OPTA_STRENGTH_DICT["eng ENG_2. Championship"]
                / FBREF_LEAGUE_OPTA_STRENGTH_DICT["eng ENG_1. Premier League"]
            )

        df_shooting: pd.DataFrame = process_stat(
            season,
            league_id,
            "shooting",
            {
                "header_standard_shots_on_target": "shots_on_target",
                "header_expected_npxg": "npxg",
                "header_standard_pens_att": "pens_won",
                "header_standard_pens_made": "pens_scored",
            },
            ["shots_on_target", "npxg", "pens_won", "pens_scored"],
        )

        df_shooting_vs: pd.DataFrame = process_stat(
            season,
            league_id,
            "shooting_against",
            {
                "header_standard_shots_on_target": "shots_on_target_vs",
                "header_expected_npxg": "npxg_vs",
            },
            ["shots_on_target_vs", "npxg_vs"],
        )

        df_passing: pd.DataFrame = process_stat(
            season,
            league_id,
            "passing",
            {
                "header_expected_pass_xa": "pass_xa",
                "assisted_shots": "key_passes",
            },
            ["pass_xa", "key_passes"],
        )

        df_gca: pd.DataFrame = process_stat(
            season,
            league_id,
            "gca",
            {
                "header_sca_sca": "sca",
                "header_gca_gca": "gca",
            },
            ["sca", "gca"],
        )

        df_misc: pd.DataFrame = process_stat(
            season,
            league_id,
            "misc",
            {
                "header_performance_cards_yellow": "yellow_cards",
                "header_performance_cards_red": "red_cards",
                "header_performance_fouls": "fouls_conceded",
                "header_performance_fouled": "fouls_won",
                "header_performance_pens_conceded": "pens_conceded",
            },
            [
                "yellow_cards",
                "red_cards",
                "fouls_conceded",
                "fouls_won",
                "pens_conceded",
            ],
        )

        df_misc_vs: pd.DataFrame = process_stat(
            season,
            league_id,
            "misc_against",
            {
                "header_performance_cards_yellow": "yellow_cards_vs",
                "header_performance_cards_red": "red_cards_vs",
            },
            ["yellow_cards_vs", "red_cards_vs"],
        )

        df_defense: pd.DataFrame = process_stat(
            season,
            league_id,
            "defense",
            {
                "header_tackles_tackles_won": "tackles_won",
                "header_blocks_blocks": "blocks",
            },
            ["tackles_won", "interceptions", "blocks", "clearances"],
        )

        df_keeper: pd.DataFrame = process_stat(
            season,
            league_id,
            "keeper",
            {
                "header_performance_gk_saves": "gk_saves",
            },
            ["gk_saves"],
            "header_playing_gk_games",
        )

        df_league: pd.DataFrame = reduce(
            lambda left, right: left.merge(
                right,
                on=["team"],
                how="left",
                validate="1:1",
            ),
            [
                df_standard,
                df_shooting,
                df_shooting_vs,
                df_passing,
                df_gca,
                df_misc,
                df_misc_vs,
                df_defense,
                df_keeper,
            ],
        )
        dfs_leagues.append(df_league)

    df_other_stats: pd.DataFrame = (
        pd.concat(dfs_leagues, ignore_index=True) if dfs_leagues else pd.DataFrame()
    )
    df_formation = df_formation.merge(
        df_other_stats, on="team", how="left", validate="1:1"
    )
    fpath: Path = DATA_FOLDER_FBREF / season.folder / "team_seasonal_stats.csv"
    save_pandas(df_formation, fpath)
    logger.info("Team seasonal averages saved for season: {}", season.fbref_long_name)


if __name__ == "__main__":
    build_team_features_prediction(Seasons.SEASON_2324.value)
