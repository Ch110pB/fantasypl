"""Functions for creating features for team models."""

from functools import reduce
from typing import TYPE_CHECKING, Literal

import pandas as pd
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.modeling_helper import (
    get_form_data,
    get_static_data,
    get_teamgw_json_to_df,
)
from fantasypl.utils.save_helper import save_pandas


if TYPE_CHECKING:
    from pathlib import Path


cols_form_for_xgoals: list[str] = [
    "possession",
    "shots_on_target",
    "npxg",
    "key_passes",
    "pass_xa",
    "sca",
    "gca",
]
cols_static_against_xgoals: list[str] = [
    "possession",
    "shots_on_target_vs",
    "npxg_vs",
    "tackles_won",
    "blocks",
    "interceptions",
    "clearances",
    "gk_saves",
]

cols_form_for_xyc: list[str] = ["yellow_cards", "red_cards", "fouls_conceded"]
cols_static_against_xyc: list[str] = [
    "yellow_cards_vs",
    "red_cards_vs",
    "fouls_won",
]

cols_form_for_xpens: list[str] = ["pens_won", "pens_scored"]
cols_static_against_xpens: list[str] = ["pens_conceded"]


def get_groups(
    data: pd.DataFrame,
    cols_form: list[str],
    cols_static: list[str],
    team_or_player: Literal["team", "opponent"],
    for_or_opp: Literal["for", "opp"],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """

    Args:
    ----
        data: A pandas dataframe having the entire dataset.
        cols_form: List of column names to get lagged features on.
        cols_static: List of column names to get aggregated features on.
        team_or_player: Team/Opponent which column to group by.
        for_or_opp: Suffix for feature.

    Returns:
    -------
        Two pandas dataframes having the lagged features and aggregated features.

    """
    grouped_form_df: pd.DataFrame = get_form_data(
        data=data,
        cols=cols_form,
        team_or_player=team_or_player,
    )
    grouped_form_df = grouped_form_df.rename(
        columns={
            **{
                col: f"{col}_{for_or_opp}"
                for col in grouped_form_df.columns
                if "_lag_" in col
            },
        },
    )
    grouped_static_df: pd.DataFrame = get_static_data(
        data=data,
        cols=cols_static,
        team_or_player=team_or_player,
    )
    grouped_static_df = grouped_static_df.rename(
        columns={
            **{
                col: f"{col}_{for_or_opp}"
                for col in grouped_static_df.columns
                if "_mean" in col
            },
        },
    )

    return grouped_form_df, grouped_static_df


def save_joined_df(  # noqa: PLR0917
    data: pd.DataFrame,
    season: Season,
    data_form_for: pd.DataFrame,
    data_static_for: pd.DataFrame,
    data_form_against: pd.DataFrame,
    data_static_against: pd.DataFrame,
    stat: str,
) -> None:
    """

    Args:
    ----
        data: A pandas dataframe having the entire dataset.
        season: Season.
        data_form_for: List of column names for lagged features for team.
        data_static_for: List of column names for aggregated features for team.
        data_form_against: List of column names for lagged features for opponent.
        data_static_against: List of column names for aggregated features for opponent.
        stat: Model name.

    """
    df_final_for: pd.DataFrame = reduce(
        lambda left, right: left.merge(
            right,
            on=["team", "date"],
            how="left",
            validate="1:1",
        ),
        [data, data_form_for, data_static_for],
    )

    df_final_against: pd.DataFrame = reduce(
        lambda left, right: left.merge(
            right,
            on=["opponent", "date"],
            how="left",
            validate="1:1",
        ),
        [data, data_form_against, data_static_against],
    )

    df_final = df_final_for.merge(
        df_final_against,
        on=list(set(df_final_for.columns) & set(df_final_against.columns)),
        how="inner",
        validate="1:1",
    )
    df_final = df_final.dropna(how="any").reset_index(drop=True)
    fpath: Path = (
        DATA_FOLDER_FBREF / season.folder / "training" / f"teams_{stat}_features.csv"
    )
    save_pandas(df=df_final, fpath=fpath)
    logger.info("Features saved for Team {}", stat)


def get_features(season: Season) -> None:
    """

    Args:
    ----
        season: Season.

    """
    team_df: pd.DataFrame = get_teamgw_json_to_df(season)
    team_df["team"] = [team.fbref_id for team in team_df["team"]]
    team_df["opponent"] = [opponent.fbref_id for opponent in team_df["opponent"]]

    save_joined_df(
        team_df,
        season,
        *get_groups(
            team_df,
            cols_form_for_xgoals,
            [],
            "team",
            "for",
        ),
        *get_groups(
            team_df,
            [],
            cols_static_against_xgoals,
            "opponent",
            "opp",
        ),
        stat="xgoals",
    )
    save_joined_df(
        team_df,
        season,
        *get_groups(
            team_df,
            cols_form_for_xyc,
            [],
            "team",
            "for",
        ),
        *get_groups(
            team_df,
            [],
            cols_static_against_xyc,
            "opponent",
            "opp",
        ),
        stat="xyc",
    )
    save_joined_df(
        team_df,
        season,
        *get_groups(
            team_df,
            cols_form_for_xpens,
            [],
            "team",
            "for",
        ),
        *get_groups(
            team_df,
            [],
            cols_static_against_xpens,
            "opponent",
            "opp",
        ),
        stat="xpens",
    )


if __name__ == "__main__":
    get_features(Seasons.SEASON_2324.value)
