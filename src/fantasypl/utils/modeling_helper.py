"""Helper functions for building ML models and predictions."""

import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer  # type: ignore[import-untyped]
from sklearn.model_selection import (  # type: ignore[import-untyped]
    train_test_split,
)
from sklearn.pipeline import Pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import-untyped]

from fantasypl.config.constants import (
    DATA_FOLDER_FBREF,
    DATA_FOLDER_REF,
    MODEL_FOLDER,
    SEED,
)
from fantasypl.config.schemas import (
    Player,
    PlayerGameWeek,
    Season,
    Team,
    TeamGameweek,
)
from fantasypl.utils import save_pkl


def get_list_teams() -> list[Team]:
    """

    Returns
    -------
        The list of Team objects from references JSON.

    """
    with Path.open(DATA_FOLDER_REF / "teams.json", "r") as f:
        list_teams: list[Team] = [
            Team.model_validate(el) for el in json.load(f).get("teams")
        ]
    return list_teams


def get_list_players() -> list[Player]:
    """

    Returns
    -------
        The list of Player objects from references JSON.

    """
    with Path.open(DATA_FOLDER_REF / "players.json", "r") as f:
        list_players: list[Player] = [
            Player.model_validate(el) for el in json.load(f).get("players")
        ]
    return list_players


def get_team_gameweek_json_to_df(season: Season) -> pd.DataFrame:
    """

    Parameters
    ----------
    season
        The season under process.

    Returns
    -------
        A pandas dataframe from the team matchlogs JSON for the season.

    """
    with Path.open(
        DATA_FOLDER_FBREF / season.folder / "team_matchlogs.json", "r"
    ) as f:
        list_team_matchlogs: list[TeamGameweek] = [
            TeamGameweek.model_validate(el)
            for el in json.load(f).get("team_matchlogs")
        ]
    return pd.DataFrame([dict(el) for el in list_team_matchlogs])


def get_player_gameweek_json_to_df(season: Season) -> pd.DataFrame:
    """

    Parameters
    ----------
    season
        The season under process.

    Returns
    -------
        A pandas dataframe from the player matchlogs JSON for the season.

    """
    with Path.open(
        DATA_FOLDER_FBREF / season.folder / "player_matchlogs.json", "r"
    ) as f:
        list_player_matchlogs: list[PlayerGameWeek] = [
            PlayerGameWeek.model_validate(el)
            for el in json.load(f).get("player_matchlogs")
        ]
    return pd.DataFrame([dict(el) for el in list_player_matchlogs])


def get_fbref_teams(season: Season) -> list[str]:
    """

    Parameters
    ----------
    season
        The season under process.

    Returns
    -------
        The list of FBRef team names for the season.

    """
    return [
        *pd.read_csv(f"{DATA_FOLDER_FBREF}/{season.folder}/teams.csv")["name"]
    ]


def get_form_data(
    data: pd.DataFrame,
    cols: list[str],
    team_or_player: Literal["team", "player", "opponent"],
) -> pd.DataFrame:
    """

    Parameters
    ----------
    data
        A pandas dataframe with all the features.
    cols
        Columns to get lagged features on.
    team_or_player
        The element to group by.

    Returns
    -------
        A pandas dataframe containing the lagged features.

    """
    data = data.sort_values(by="date", ascending=True)
    for col in cols:
        shifted: pd.DataFrame = (
            data.groupby(team_or_player)[col]
            .shift(range(1, 6))
            .pivot_table()
            .add_prefix(f"{col}_lag_")
        )
        data = pd.concat([data, shifted], axis=1)
    return data[
        [
            team_or_player,
            "date",
            *[col for col in data.columns if "_lag_" in col],
        ]
    ]


def get_static_data(
    data: pd.DataFrame,
    cols: list[str],
    team_or_player: Literal["team", "player", "opponent"],
) -> pd.DataFrame:
    """

    Parameters
    ----------
    data
        A pandas dataframe with all the features.
    cols
        Columns to get aggregated features on.
    team_or_player
        The element to group by.

    Returns
    -------
        A pandas dataframe containing the aggregated features.

    """
    data = data.sort_values(by="date", ascending=True)
    for col in cols:
        data[f"{col}_mean"] = (
            data.groupby(team_or_player)[col].shift(1).rolling(window=5).mean()
        )
    return data[
        [
            team_or_player,
            "date",
            *[col for col in data.columns if "_mean" in col],
        ]
    ]


def preprocess_data_and_save(  # noqa: PLR0913, PLR0917
    df: pd.DataFrame,
    target_col: str,
    target_name: str,
    categorical_features: list[str],
    categories: list[list[str]],
    team_or_player: Literal["team", "player"],
    season: Season,
    position: str | None = None,
) -> None:
    """

    Parameters
    ----------
    df
        The full dataframe.
    target_col
        The target(y) column.
    target_name
        The model name.
    categorical_features
        List of columns with categorical features.
    categories
        List of categories for the categorical features.
    team_or_player
        The element to create models for
    season
        The season under process.
    position
        The short_position of player to create models for.
        None for team models.

    """
    df_us: pd.DataFrame = (
        pd.concat(
            [
                df[df[target_col] != 0],
                df[df[target_col] == 0].sample(frac=0.3, random_state=SEED),
            ],
            ignore_index=True,
        )
        if not df.empty
        else pd.DataFrame()
    )

    categorical_transformer: Pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(categories=categories, drop="first"))],
        memory=None,
    )
    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[("cat", categorical_transformer, categorical_features)],
        remainder="passthrough",
    )
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    df_train, df_test = train_test_split(
        df_us, train_size=0.8, random_state=SEED, shuffle=True
    )
    x_train: pd.DataFrame
    y_train: npt.NDArray[np.float32]
    x_test: pd.DataFrame
    y_test: npt.NDArray[np.float32]
    x_train, y_train = (
        df_train.drop(columns=target_col),
        df_train[target_col].to_numpy(dtype=np.float32),
    )
    x_test, y_test = (
        df_test.drop(columns=target_col),
        df_test[target_col].to_numpy(dtype=np.float32),
    )
    x_train_np: npt.NDArray[np.float32] = preprocessor.fit_transform(x_train)
    x_test_np: npt.NDArray[np.float32] = preprocessor.transform(x_test)

    dict_array: dict[str, npt.NDArray[np.float32]] = {
        "x_train": x_train_np,
        "y_train": y_train,
        "x_test": x_test_np,
        "y_test": y_test,
    }

    folder: str = (
        season.folder if position is None else f"{season.folder}/{position}"
    )
    for key, value in dict_array.items():
        fpath: Path = (
            MODEL_FOLDER
            / folder
            / f"model_{team_or_player}_{target_name}/{key}.pkl"
        )
        save_pkl(obj=value, fpath=fpath)
    fpath_preproc: Path = (
        MODEL_FOLDER
        / folder
        / f"model_{team_or_player}_{target_name}/preprocessor.pkl"
    )
    save_pkl(obj=preprocessor, fpath=fpath_preproc)


def get_train_test_data(
    folder: str, season: Season
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """

    Parameters
    ----------
    folder
        The folder containing train-test splits.
    season
        The season under process.

    Returns
    -------
        The train-test splits.

    """
    list_array: list[str] = ["x_train", "y_train", "x_test", "y_test"]
    dict_array: dict[str, npt.NDArray[np.float32]] = {}
    for arr in list_array:
        with Path.open(
            MODEL_FOLDER / season.folder / folder / f"{arr}.pkl", "rb"
        ) as f:
            dict_array[arr] = pickle.load(f)
    return (
        dict_array["x_train"],
        dict_array["y_train"],
        dict_array["x_test"],
        dict_array["y_test"],
    )
