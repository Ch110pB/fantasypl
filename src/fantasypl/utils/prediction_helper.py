"""Helper functions for current season predictions."""

import pandas as pd


def pad_lists(row: pd.Series, df2: pd.DataFrame, col: str) -> pd.Series:  # type: ignore[type-arg]
    """

    Args:
    ----
        row: A row of running dataframe.
        df2: The aggregate dataframe.
        col: Column name.

    Returns:
    -------
        The modified row.

    """
    value: str | int | float = df2.at[row["team"], col]  # noqa: PD008
    if len(row[col]) < 5:  # noqa: PLR2004
        row[col] = [value] * (5 - len(row[col])) + row[col]
    return row[col]  # type: ignore[no-any-return]
