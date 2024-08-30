"""Helper functions for building models and predictions."""

import json
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fantasypl.config.constants.folder_config import DATA_FOLDER_FBREF, MODEL_FOLDER
from fantasypl.config.constants.modeling_config import SEED
from fantasypl.config.models.season import Season
from fantasypl.config.models.team_gameweek import TeamGameweek
from fantasypl.utils.save_helper import save_pkl


def get_team_gameweek_json_to_df(season: Season) -> pd.DataFrame:
    """
    Args:
    ----
        season: Season.

    Returns:
    -------
        A pandas dataframe converted from the team matchlogs JSON for the season.

    """
    with Path.open(DATA_FOLDER_FBREF / season.folder / "team_matchlogs.json", "r") as f:
        list_team_matchlogs: list[TeamGameweek] = [
            TeamGameweek.model_validate(el) for el in json.load(f).get("team_matchlogs")
        ]
    return pd.DataFrame([dict(el) for el in list_team_matchlogs])


def get_fbref_teams(season: Season) -> list[str]:
    """
    Args:
    ----
        season: Season.

    Returns:
    -------
        A list of team names for the season.

    """
    return pd.read_csv(f"{DATA_FOLDER_FBREF}/{season.folder}/teams.csv")[
        "name"
    ].tolist()


def get_form_data(
    data: pd.DataFrame,
    cols: list[str],
    team_or_player: Literal["team", "player", "opponent"],
) -> pd.DataFrame:
    """
    Args:
    ----
        data: A pandas dataframe having the entire dataset.
        cols: List of column names to get lagged features on.
        team_or_player: Team/Player - the element to group by.

    Returns:
    -------
        A pandas dataframe containing the lagged features.

    """
    data = data.sort_values(by="date", ascending=True)
    for col in cols:
        for i in range(1, 6):
            data[f"{col}_lag_{i}"] = data.groupby(team_or_player)[col].shift(i)
    return data[
        [team_or_player, "date", *[col for col in data.columns if "_lag_" in col]]
    ]


def get_static_data(
    data: pd.DataFrame,
    cols: list[str],
    team_or_player: Literal["team", "player", "opponent"],
) -> pd.DataFrame:
    """
    Args:
    ----
        data: A pandas dataframe having the entire dataset.
        cols: List of column names to get aggregated features on.
        team_or_player: Team/Player - the element to group by.

    Returns:
    -------
        A pandas dataframe containing the aggregated features.

    """
    data = data.sort_values(by="date", ascending=True)
    for col in cols:
        data[f"{col}_mean"] = (
            data.groupby(team_or_player)[col].shift(1).rolling(window=5).mean()
        )
    return data[
        [team_or_player, "date", *[col for col in data.columns if "_mean" in col]]
    ]


def preprocess_data_and_save(  # noqa: PLR0917
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
    Args:
    ----
        df: The entire features dataframe.
        target_col: The target(y) column.
        target_name: The model name.
        categorical_features: List of columns containing categorical features.
        categories: List of all categories of the categorical columns.
        team_or_player: Team/Player - the element to create features for.
        season: Season.
        position: FBRef position for player models. None for team models.

    """
    df_us: pd.DataFrame = (
        pd.concat(
            [
                df[df[target_col] != 0],
                df[df[target_col] == 0].sample(frac=0.5, random_state=SEED),
            ],
            ignore_index=True,
        )
        if not df.empty
        else pd.DataFrame()
    )

    categorical_transformer: Pipeline = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(categories=categories, drop="first")),
        ],
        memory=None,
    )
    preprocessor: ColumnTransformer = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="passthrough",
    )
    df_train: pd.DataFrame
    df_test: pd.DataFrame
    df_train, df_test = train_test_split(
        df_us,
        train_size=0.8,
        random_state=SEED,
        shuffle=True,
    )
    x_train: pd.DataFrame
    y_train: npt.NDArray[np.float32]
    x_test: pd.DataFrame
    y_test: npt.NDArray[np.float32]
    x_train, y_train = (
        df_train.drop(columns=target_col).copy(),
        df_train[target_col].copy().to_numpy(dtype=np.float32),
    )
    x_test, y_test = (
        df_test.drop(columns=target_col).copy(),
        df_test[target_col].copy().to_numpy(dtype=np.float32),
    )
    x_train_np: npt.NDArray[np.float32] = preprocessor.fit_transform(x_train)
    x_test_np: npt.NDArray[np.float32] = preprocessor.transform(x_test)

    dict_array: dict[str, npt.NDArray[np.float32]] = {
        "x_train": x_train_np,
        "y_train": y_train,
        "x_test": x_test_np,
        "y_test": y_test,
    }

    folder: str = season.folder if position is None else f"{season.folder}/{position}"
    for key, value in dict_array.items():
        fpath: Path = (
            MODEL_FOLDER / folder / f"model_{team_or_player}_{target_name}/{key}.pkl"
        )
        save_pkl(value, fpath)
    fpath_preproc: Path = (
        MODEL_FOLDER / folder / f"model_{team_or_player}_{target_name}/preprocessor.pkl"
    )
    save_pkl(obj=preprocessor, fpath=fpath_preproc)


def get_train_test_data(
    folder: str,
    season: Season,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    """
    Args:
    ----
        folder: The folder to get train-test splits.
        season: Season.

    Returns:
    -------
        The train and test splits.

    """
    list_array: list[str] = ["x_train", "y_train", "x_test", "y_test"]
    dict_array: dict[str, npt.NDArray[np.float32]] = {}
    for arr in list_array:
        with Path.open(MODEL_FOLDER / season.folder / folder / f"{arr}.pkl", "rb") as f:
            dict_array[arr] = pickle.load(f)
    return (
        dict_array["x_train"],
        dict_array["y_train"],
        dict_array["x_test"],
        dict_array["y_test"],
    )
