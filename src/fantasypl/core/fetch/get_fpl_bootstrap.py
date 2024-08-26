from typing import TYPE_CHECKING

import requests
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FPL
from fantasypl.config.constants.web_config import FPL_BOOTSTRAP_URL, FPL_FIXTURES_URL
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.save_helper import save_json


if TYPE_CHECKING:
    from pathlib import Path


def get_bootstrap(season: Season) -> None:
    response: requests.Response = requests.get(FPL_BOOTSTRAP_URL)
    fpath: Path = DATA_FOLDER_FPL / season.folder / "bootstrap.json"
    save_json(response.json(), fpath)
    logger.info("FPL Bootstrap downloaded for season {}", season.fbref_name)


def get_fixtures(season: Season) -> None:
    response: requests.Response = requests.get(FPL_FIXTURES_URL)
    fpath: Path = DATA_FOLDER_FPL / season.folder / "fixtures.json"
    save_json(response.json(), fpath)
    logger.info("FPL Fixtures downloaded for season {}", season.fbref_name)


if __name__ == "__main__":
    get_bootstrap(Seasons.SEASON_2425.value)
    get_fixtures(Seasons.SEASON_2425.value)
