import pandas as pd
from loguru import logger

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF
from fantasypl.config.models.season import Season, Seasons
from fantasypl.utils.modeling_helper import preprocess_data_and_save


def build_split(season: Season, target_name: str, target_col: str) -> None:
    df: pd.DataFrame = pd.read_csv(
        DATA_FOLDER_FBREF
        / season.folder
        / "training"
        / f"teams_{target_name}_features.csv",
    )
    _select_cols: list[str] = [
        col
        for col in df.columns
        if ("_lag_" in col) or ("_mean_" in col) or (col == "venue")
    ]
    _add_select_cols: list[str]
    match target_name:
        case "xgoals":
            _add_select_cols = [target_col, "formation", "formation_vs"]
        case "xyc" | "xpens":
            _add_select_cols = [target_col]
        case _:
            _add_select_cols = []

    df_pd: pd.DataFrame = df[_select_cols + _add_select_cols]
    categorical_features: list[str]
    match target_name:
        case "xgoals":
            categorical_features = ["venue", "formation", "formation_vs"]
        case _:
            categorical_features = ["venue"]
    categories: list[list[str]] = [
        df_pd[feature].unique().tolist() for feature in categorical_features
    ]
    preprocess_data_and_save(
        df=df_pd,
        target_col=target_col,
        target_name=target_name,
        categorical_features=categorical_features,
        categories=categories,
        team_or_player="team",
        season=season,
    )
    logger.info("Train-test splits and preprocessor saved for team {}", target_name)


if __name__ == "__main__":
    build_split(Seasons.SEASON_2324.value, "xgoals", "npxg")
    build_split(Seasons.SEASON_2324.value, "xyc", "yellow_cards")
    build_split(Seasons.SEASON_2324.value, "xpens", "pens_scored")
