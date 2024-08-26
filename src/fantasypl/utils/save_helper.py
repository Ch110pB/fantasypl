import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(
    json_dict: dict[str, Any],
    fpath: Path,
    default: Any | None = None,
) -> None:
    Path.mkdir(fpath.parent, parents=True, exist_ok=True)
    with Path.open(fpath, "w") as f:
        json.dump(json_dict, f, default=default)


def save_pandas(df: pd.DataFrame, fpath: Path) -> None:
    Path.mkdir(fpath.parent, parents=True, exist_ok=True)
    df.to_csv(fpath, index=False)


def save_pkl(obj: Any, fpath: Path, protocol: int | None = None) -> None:
    Path.mkdir(fpath.parent, parents=True, exist_ok=True)
    with Path.open(fpath, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)
