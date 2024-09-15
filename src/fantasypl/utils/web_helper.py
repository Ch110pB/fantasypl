"""Helper functions for scraping the web."""

import asyncio
import operator
import time
from functools import reduce
from typing import Awaitable

import pandas as pd
import requests
from lxml import html


def get_content(url: str, delay: int = 8, timeout: int = 15) -> str:
    """
    Get the contents of a web page.

    Parameters
    ----------
    url
        The URL to scrape.
    delay
        The delay between requests in seconds.
    timeout
        The timeout in seconds.

    Returns
    -------
        The contents of the web page.

    """
    with requests.Session() as session:
        response: requests.models.Response = session.get(
            url=url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=timeout,
        )
        time.sleep(delay)
    return response.content.decode("utf-8")


def extract_table(
    content: str,
    table_id: str,
    *,
    href: bool = False,
    dropna_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract the table from the web page content given the table ID.

    Parameters
    ----------
    content
        The contents of the web page.
    table_id
        The table ID to fetch.
    href
        Boolean value for whether href links are required.
    dropna_cols
        Columns to mark NA rows.

    Returns
    -------
        A pandas dataframe containing the table data.

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
    *,
    href: bool = False,
    dropna_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Extract the table asynchronously.

    Returns
    -------
        A pandas dataframe containing the table data.

    """
    return await asyncio.to_thread(
        extract_table,
        content,
        table_id,
        href=href,
        dropna_cols=dropna_cols,
    )


async def get_single_table(
    content: str,
    tables: list[str],
    *,
    href: bool = False,
    dropna_cols: list[str] | None = None,
) -> list[pd.DataFrame]:
    """
    Merge the results into a list of pandas dataframes.

    Returns
    -------
        A list of pandas dataframes containing table data.

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
