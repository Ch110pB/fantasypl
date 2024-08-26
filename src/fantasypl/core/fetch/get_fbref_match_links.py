from typing import TYPE_CHECKING

from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF
from fantasypl.config.constants.web_config import FBREF_BASE_URL
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.save_helper import save_pandas
from fantasypl.utils.web_helper import extract_table, get_content


if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


def get_match_links(season: Season) -> None:
    url: str = f"{FBREF_BASE_URL}/comps/9/{season.fbref_long_name}/schedule/"
    content: str = get_content(url=url, delay=0)
    table_id: str = f"sched_{season.fbref_long_name}_9_1"
    df_links: pd.DataFrame = extract_table(
        content=content,
        table_id=table_id,
        href=True,
        dropna_cols=["score"],
    )
    if df_links.empty:
        logger.error("FBRef Data fetch failed for Season: {}", season.fbref_name)
        return
    df_links = df_links[["date", "home_team", "score", "away_team"]]
    df_links["home_team"] = (
        df_links["home_team"].str[1].str.strip().str.split("/").str[3]
    )
    df_links["away_team"] = (
        df_links["away_team"].str[1].str.strip().str.split("/").str[3]
    )
    df_links["match_link"] = df_links["score"].str[1]
    df_links["date"] = df_links["date"].str[0]
    df_links = df_links.drop(columns="score").loc[df_links["match_link"] != ""]
    fpath: Path = DATA_FOLDER_FBREF / season.folder / "match_links.csv"
    save_pandas(df_links, fpath)
    logger.info(
        "Match links data saved for Season: {} | Total Matches: {}",
        season.fbref_name,
        df_links.shape[0],
    )


if __name__ == "__main__":
    # get_match_links(Seasons.SEASON_2324.value)
    get_match_links(Seasons.SEASON_2425.value)
