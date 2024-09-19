"""Functions to calculate expected stats for players for each gameweek."""

from functools import reduce
from typing import TYPE_CHECKING

import pandas as pd
from loguru import logger

from fantasypl.config.constants import MODEL_FOLDER
from fantasypl.utils import save_pandas


if TYPE_CHECKING:
    from pathlib import Path


def calc_final_stats(gameweek: int) -> None:
    """
    Calculate expected stats for the players for the gameweek.

    Parameters
    ----------
    gameweek
        The gameweek under process.

    """
    team_preds_path: Path = (
        MODEL_FOLDER / "predictions/team" / f"gameweek_{gameweek}"
    )
    player_preds_path: Path = (
        MODEL_FOLDER / "predictions/player" / f"gameweek_{gameweek}"
    )

    df_team_predictions: pd.DataFrame = reduce(
        lambda left, right: left.merge(
            right,
            on=["team", "opponent", "gameweek"],
            how="left",
            validate="1:1",
        ),
        [
            pd.read_csv(team_preds_path / f"prediction_{model}.csv")
            for model in ["xgoals", "xpens", "xyc"]
        ],
    )
    df_team_predictions = df_team_predictions.rename(
        columns={**{col: f"team_{col}" for col in ["xgoals", "xpens", "xyc"]}},
    )

    dict_xgoals: dict[tuple[str, int], float] = (
        df_team_predictions[["team", "gameweek", "team_xgoals"]]
        .set_index(["team", "gameweek"])
        .to_dict()["team_xgoals"]
    )
    df_team_predictions["xgoals_vs"] = [
        dict_xgoals[opponent, gameweek]
        for opponent, gameweek in zip(
            df_team_predictions["opponent"], df_team_predictions["gameweek"]
        )
    ]

    dfs_player: list[pd.DataFrame] = []
    for pos in ["GK", "DF", "MF", "FW"]:
        dfs: list[pd.DataFrame] = []
        for model in ["xgoals", "xassists", "xmins", "xpens", "xyc", "xsaves"]:
            try:
                df_: pd.DataFrame = pd.read_csv(
                    player_preds_path / pos / f"prediction_{model}.csv",
                )
                dfs.append(df_)
            except FileNotFoundError:  # noqa: PERF203
                pass
        df_position: pd.DataFrame = reduce(
            lambda left, right: left.merge(
                right,
                on=["player", "team", "gameweek", "short_position"],
                how="left",
                validate="m:m",
            ),
            dfs,
        )
        dfs_player.append(df_position)
    df_player_predictions: pd.DataFrame = (
        pd.concat(dfs_player) if dfs_player else pd.DataFrame()
    )

    df_expected_stats: pd.DataFrame = df_player_predictions.merge(
        df_team_predictions,
        on=["team", "gameweek"],
        how="left",
        validate="m:m",
    )
    df_expected_stats = df_expected_stats.fillna(0)

    df_expected_stats["xgoals"] = (
        df_expected_stats["xgoals"]
        * df_expected_stats["team_xgoals"]
        / df_expected_stats.groupby(["team", "gameweek"])["xgoals"].transform(
            "sum",
        )
    )
    df_expected_stats["xassists"] = (
        df_expected_stats["xassists"]
        * df_expected_stats["team_xgoals"]
        / df_expected_stats.groupby(["team", "gameweek"])[
            "xassists"
        ].transform("sum")
    )
    df_expected_stats["xmins"] = (
        df_expected_stats["xmins"]
        * 990
        / df_expected_stats.groupby(["team", "gameweek"])["xmins"].transform(
            "sum",
        )
    )
    df_expected_stats["xyc"] = (
        df_expected_stats["xyc"]
        * df_expected_stats["team_xyc"]
        / df_expected_stats.groupby(["team", "gameweek"])["xyc"].transform(
            "sum",
        )
    )
    df_expected_stats["xpens"] = (
        df_expected_stats["xpens"]
        * df_expected_stats["team_xpens"]
        / df_expected_stats.groupby(["team", "gameweek"])["xpens"].transform(
            "sum",
        )
    )

    df_expected_stats = df_expected_stats[
        [
            "player",
            "team",
            "gameweek",
            "xgoals",
            "xassists",
            "xmins",
            "xyc",
            "xsaves",
            "xpens",
            "xgoals_vs",
        ]
    ]
    save_pandas(
        df_expected_stats,
        player_preds_path / "prediction_expected_stats.csv",
    )
    logger.info("Expected stats saved for all players.")


if __name__ == "__main__":
    calc_final_stats(5)
