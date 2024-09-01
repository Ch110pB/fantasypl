"""Functions for getting FBRef player stats for all seasons."""

import asyncio
from typing import TYPE_CHECKING

import rich.progress
from loguru import logger
from lxml import html

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF
from fantasypl.config.constants.web_config import FBREF_BASE_URL
from fantasypl.config.models.season import Season, Seasons
from fantasypl.config.references.player_refs import FBREF_FPL_PLAYER_REF_DICT
from fantasypl.utils.save_helper import save_json, save_pandas
from fantasypl.utils.web_helper import get_content, get_single_table


if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


_tables: list[str] = [
    "stats_standard_dom_lg",
    "stats_playing_time_dom_lg",
    "stats_shooting_dom_lg",
    "stats_passing_dom_lg",
    "stats_defense_dom_lg",
    "stats_gca_dom_lg",
    "stats_misc_dom_lg",
    "stats_keeper_dom_lg",
    "stats_keeper_adv_dom_lg",
]


def get_player_season(season: Season, filter_players: list[str] | None = None) -> None:
    """

    Args:
    ----
        season: Season, folder to save the data.
        filter_players: Optional list of FBRef IDs to run on.

    """
    list_players: list[str] = [*FBREF_FPL_PLAYER_REF_DICT]
    if filter_players is not None:
        list_players = filter_players.copy()
    for player_id in rich.progress.track(
        list_players,
        description="Getting player pages from FBRef: ",
    ):
        try:
            content: str = get_content(f"{FBREF_BASE_URL}/players/{player_id}/")
            tree: html.HtmlElement = html.fromstring(content)
            infobox: html.HtmlElement = next(
                el
                for el in tree.cssselect("div#meta")
                if el.get("class") != "media-item"
            )
            fbref_name: str = infobox.cssselect("h1")[0].text_content().strip()
            position: str = (
                next(
                    el
                    for el in infobox.cssselect("p")
                    if "Position" in el.text_content()
                )
                .text_content()
                .split("▪")[0]
                .strip()
                .replace("Position: ", "")
            )
            dfs: list[pd.DataFrame] = asyncio.run(
                get_single_table(content=content, tables=_tables),
            )
            for j, df in enumerate(dfs):
                fpath: Path = (
                    DATA_FOLDER_FBREF
                    / season.folder
                    / "player_season"
                    / f"{player_id}_{
                        _tables[j].removeprefix("stats_").removesuffix("_dom_lg")
                    }.csv"
                )

                if df.empty and (
                    not (
                        (position == "GK" and "keeper" not in _tables[j])
                        or (position != "GK" and "keeper" in _tables[j])
                    )
                ):
                    logger.error(
                        "Data fetch error from FBRef: "
                        "Season = {} Player ID = {} "
                        "Stat = {}",
                        season.fbref_name,
                        player_id,
                        f"{player_id}_{
                            _tables[j].removeprefix("stats_").removesuffix("_dom_lg")
                        }",
                    )
                save_pandas(df=df, fpath=fpath)
            df_details: dict[str, str] = {"name": fbref_name, "position": position}
            save_json(
                df_details,
                (
                    DATA_FOLDER_FBREF
                    / season.folder
                    / "player_season"
                    / f"{player_id}.json"
                ),
            )
        except StopIteration:
            logger.error("Fetching page failed for Player ID: {}", player_id)


if __name__ == "__main__":
    get_player_season(
        Seasons.SEASON_2324.value,
        filter_players=["6a10efb3", "6c2b66fc"],
    )
