"""Helper functions for scraping the web for stats."""

import asyncio
import operator
import time
from functools import reduce
from typing import Awaitable

import pandas as pd
import requests
from lxml import html


def get_content(url: str, delay: int = 8) -> str:
    """
    Args:
    ----
        url: URL to scrape.
        delay: Delay between GET calls in seconds.

    Returns:
    -------
        The webpage content.

    """
    with requests.Session() as session:
        response: requests.models.Response = session.get(
            url=url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        time.sleep(delay)
    return response.content.decode("utf-8")


def extract_table(
    content: str,
    table_id: str,
    href: bool = False,
    dropna_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Args:
    ----
        content: The webpage content.
        table_id: The table ID to scrape.
        href: True if some links are needed, False otherwise.
        dropna_cols: Columns to mark empty rows.

    Returns:
    -------
        A pandas dataframe containing data from the table.

    """
    if dropna_cols is None:
        dropna_cols = []
    tree: html.HtmlElement = html.fromstring(content)
    try:
        table: html.HtmlElement = tree.cssselect(f"table#{table_id}")[0]
    except IndexError:
        return pd.DataFrame()
    headers_multi_index: list[list[str]] = [
        reduce(
            operator.add,
            [
                [cell.get("data-stat")] * int(cell.get("colspan", 1))
                for cell in row.cssselect("th")
            ],
        )
        for row in table.cssselect("thead>tr")
    ]
    headers: list[str] = [
        "_".join(filter(None, items))
        for items in zip(*headers_multi_index, strict=False)
    ]

    df_table: pd.DataFrame
    if href:
        table_data: list[list[tuple[str, str]]] = [
            [
                (cell.text_content(), cell.cssselect("a")[0].get("href"))
                if cell.cssselect("a")
                else (cell.text_content(), "")
                for cell in row.cssselect("td, th")
            ]
            for row in table.cssselect("tbody>tr")
            if len(row.cssselect("td"))
        ]
        df_table = pd.DataFrame(table_data, columns=headers)
    else:
        table_data_simple: list[list[str]] = [
            [cell.text_content() for cell in row.cssselect("td, th")]
            for row in table.cssselect("tbody>tr")
            if len(row.cssselect("td"))
        ]
        df_table = pd.DataFrame(table_data_simple, columns=headers)
    if dropna_cols:
        df_table = df_table.dropna(subset=dropna_cols, how="any")
        df_table = df_table.loc[~df_table[dropna_cols].eq("").any(axis=1)]
    return df_table


async def extract_table_async(
    content: str,
    table_id: str,
    href: bool = False,
    dropna_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Asynchronous version of extract_table().

    Returns
    -------
        A pandas dataframe containing data from the table.

    """
    return await asyncio.to_thread(extract_table, content, table_id, href, dropna_cols)


async def get_single_table(
    content: str,
    tables: list[str],
    href: bool = False,
    dropna_cols: list[str] | None = None,
) -> list[pd.DataFrame]:
    """
    Merges the results from extract table async into a list of pandas dataframes.

    Returns
    -------
        A list of pandas dataframes containing data from the table.

    """
    coroutines: list[Awaitable[pd.DataFrame]] = [
        extract_table_async(
            content=content,
            table_id=table_id,
            href=href,
            dropna_cols=dropna_cols,
        )
        for table_id in tables
    ]
    return list(await asyncio.gather(*coroutines))
