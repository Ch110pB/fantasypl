import pandas as pd
from loguru import logger

from fantasypl.config.constants.folder_config import (
    DATA_FOLDER_FBREF,
    DATA_FOLDER_FPL,
    DATA_FOLDER_REF,
)
from fantasypl.config.models.season import Seasons
from fantasypl.config.models.team import Team
from fantasypl.config.references.team_refs import FBREF_FPL_TEAM_REF_DICT
from fantasypl.utils.save_helper import save_json


def get_team_references() -> None:
    teams: list[Team] = []
    for season in [Seasons.SEASON_2324.value, Seasons.SEASON_2425.value]:
        df_fpl_teams: pd.DataFrame = pd.read_csv(
            DATA_FOLDER_FPL / season.folder / "teams.csv",
        )
        df_fbref_teams: pd.DataFrame = pd.read_csv(
            DATA_FOLDER_FBREF / season.folder / "teams.csv",
        )

        df_fpl_teams = df_fpl_teams.drop(columns="id").rename(
            columns={"code": "fpl_code", "name": "fpl_name"},
        )
        df_fbref_teams = df_fbref_teams.rename(columns={"name": "fbref_name"})
        df_fbref_teams["fpl_code"] = df_fbref_teams["fbref_id"].map(
            FBREF_FPL_TEAM_REF_DICT,
        )
        df_fbref_teams = df_fbref_teams.merge(
            df_fpl_teams,
            on="fpl_code",
            how="right",
            validate="1:1",
        )
        teams += [
            Team.model_validate(row) for row in df_fbref_teams.to_dict(orient="records")
        ]
    teams_str: list[dict[str, Team]] = [team.model_dump() for team in list(set(teams))]
    save_json({"teams": teams_str}, DATA_FOLDER_REF / "teams.json")
    logger.info("Teams class object references JSON successfully saved")


if __name__ == "__main__":
    get_team_references()
