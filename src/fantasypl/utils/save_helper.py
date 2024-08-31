"""Helper functions for saving objects."""

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import requests


def save_json(
    json_dict: dict[str, Any],
    fpath: Path,
    default: Any | None = None,  # noqa: ANN401
) -> None:
    """
    Args:
    ----
        json_dict: The dictionary to save.
        fpath: The path to save in.
        default: The default parameter for json.dump().

    """
    Path.mkdir(fpath.parent, parents=True, exist_ok=True)
    with Path.open(fpath, "w") as f:
        json.dump(json_dict, f, default=default)


def save_pandas(df: pd.DataFrame, fpath: Path) -> None:
    """
    Args:
    ----
        df: The pandas dataFrame to save.
        fpath: The path to save in.

    """
    Path.mkdir(fpath.parent, parents=True, exist_ok=True)
    df.to_csv(fpath, index=False)


def save_pkl(obj: Any, fpath: Path, protocol: int | None = None) -> None:  # noqa: ANN401
    """
    Args:
    ----
        obj: The object to save.
        fpath: The path to save in.
        protocol: The protocol parameter for pickle.dump().

    """
    Path.mkdir(fpath.parent, parents=True, exist_ok=True)
    with Path.open(fpath, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)


def save_requests_response(response: requests.Response, fpath: Path) -> None:
    """

    Args:
    ----
        response: Response from URL.
        fpath: The path to save in.

    """
    Path.mkdir(fpath.parent, parents=True, exist_ok=True)
    with Path.open(fpath, "wb") as f:
        f.write(response.content)
