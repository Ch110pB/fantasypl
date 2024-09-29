"""Run all the functions from a single place."""

import json

import pandas as pd
from loguru import logger

from fantasypl.config.schemas import Seasons
from fantasypl.core.fetch.get_fbref_match_links import get_match_links
from fantasypl.core.fetch.get_fbref_matches import get_matches
from fantasypl.core.fetch.get_fbref_player_last_season import get_player_season
from fantasypl.core.fetch.get_fbref_team_matchlogs import get_matchlogs
from fantasypl.core.fetch.get_fpl_bootstrap import get_bootstrap, get_fixtures
from fantasypl.core.fetch.get_fpl_team_data import (
    get_all_transfers,
    get_current_team,
)
from fantasypl.core.predict.calc_gameweek_final_predictions import (
    calc_final_stats,
)
from fantasypl.core.predict.calc_gameweek_matches import get_gw_matches
from fantasypl.core.predict.calc_gameweek_xpoints import calc_xpoints
from fantasypl.core.predict.calc_predict_player_features import (
    build_predict_features_player,
    predict_for_stat_player,
)
from fantasypl.core.predict.calc_predict_team_features import (
    build_predict_features_team,
    predict_for_stat_team,
)
from fantasypl.core.predict.find_optimal_squad import find_squad
from fantasypl.core.predict.find_optimal_transfers import (
    find_optimal_transfers,
)
from fantasypl.core.predict.process_last_season_player_averages import (
    build_players_features_prediction,
)
from fantasypl.core.process.process_refs_player import get_player_references
from fantasypl.core.process.save_fbref_agg_player_matchlogs import (
    save_aggregate_player_matchlogs,
)
from fantasypl.core.process.save_fbref_agg_team_matchlogs import (
    save_aggregate_team_matchlogs,
)
from fantasypl.core.process.save_fpl_teams_players import save_players
from fantasypl.utils import (
    build_fpl_lineup,
    prepare_pitch,
    prepare_transfers,
    send_discord_message,
)


if __name__ == "__main__":
    gameweek: int = int(input("Enter gameweek: "))
    team_id: int = 85599

    get_bootstrap(Seasons.SEASON_2425.value)
    get_fixtures(Seasons.SEASON_2425.value)

    save_players(Seasons.SEASON_2425.value)
    filter_players: list[str] = json.loads(
        input("Enter a list of strings of FBRef player IDs to add: ")
    )
    get_player_season(
        Seasons.SEASON_2324.value,
        filter_players=filter_players,
    )
    get_player_references(Seasons.SEASON_2324.value)
    build_players_features_prediction(
        Seasons.SEASON_2324.value, Seasons.SEASON_2425.value
    )

    get_match_links(Seasons.SEASON_2425.value)
    last_deadline_date: str = input(
        "Enter the FPL deadline date for last gameweek: "
    )
    get_matches(Seasons.SEASON_2425.value, last_deadline_date)
    get_matchlogs(Seasons.SEASON_2425.value)

    save_aggregate_team_matchlogs(Seasons.SEASON_2425)
    save_aggregate_player_matchlogs(Seasons.SEASON_2425)

    get_gw_matches(Seasons.SEASON_2425.value, gameweek)

    df_features_team: pd.DataFrame = build_predict_features_team(
        Seasons.SEASON_2425.value, gameweek
    )
    predict_for_stat_team(df_features_team, "xgoals", gameweek)
    predict_for_stat_team(df_features_team, "xyc", gameweek)
    predict_for_stat_team(df_features_team, "xpens", gameweek)

    df_features_player: pd.DataFrame = build_predict_features_player(
        Seasons.SEASON_2425.value,
        gameweek,
        Seasons.SEASON_2324.value,
    )
    for pos_ in ["GK"]:
        predict_for_stat_player(
            df_features_player,
            pos_,
            "xsaves",
            gameweek,
            Seasons.SEASON_2324.value,
        )
    for pos_ in ["MF", "FW"]:
        predict_for_stat_player(
            df_features_player,
            pos_,
            "xpens",
            gameweek,
            Seasons.SEASON_2324.value,
        )
    for pos_ in ["DF", "MF", "FW"]:
        predict_for_stat_player(
            df_features_player,
            pos_,
            "xgoals",
            gameweek,
            Seasons.SEASON_2324.value,
        )
        predict_for_stat_player(
            df_features_player,
            pos_,
            "xassists",
            gameweek,
            Seasons.SEASON_2324.value,
        )
    for pos_ in ["GK", "DF", "MF", "FW"]:
        predict_for_stat_player(
            df_features_player,
            pos_,
            "xmins",
            gameweek,
            Seasons.SEASON_2324.value,
        )
        predict_for_stat_player(
            df_features_player,
            pos_,
            "xyc",
            gameweek,
            Seasons.SEASON_2324.value,
        )

    calc_final_stats(gameweek)
    calc_xpoints(gameweek, Seasons.SEASON_2425.value)

    eleven, subs, cap = find_squad(gameweek)
    logger.info("Starting Lineup: {}", eleven)
    logger.info("Bench: {}", subs)
    logger.info("Captain: {}", cap)
    eleven_players = build_fpl_lineup(eleven, Seasons.SEASON_2425.value)
    sub_players = build_fpl_lineup(subs, Seasons.SEASON_2425.value)
    pitch = prepare_pitch(
        eleven_players, sub_players, cap, Seasons.SEASON_2425.value
    )
    message: str = "**Optimal Squad**"
    send_discord_message(message, [pitch])

    get_all_transfers(team_id, gameweek)
    get_current_team(team_id, gameweek - 1)

    eleven, subs, cap, out, ft, hit = find_optimal_transfers(
        gameweek, Seasons.SEASON_2425.value
    )
    logger.info("Starting Lineup: {}", eleven)
    logger.info("Bench: {}", subs)
    logger.info("Captain: {}", cap)
    logger.info("Out: {}", out)
    logger.info("In on FT: {}", ft)
    logger.info("In on Hit: {}", hit)

    eleven_players = build_fpl_lineup(eleven, Seasons.SEASON_2425.value)
    sub_players = build_fpl_lineup(subs, Seasons.SEASON_2425.value)
    pitch = prepare_pitch(
        eleven_players, sub_players, cap, Seasons.SEASON_2425.value
    )
    if len(out) >= 0:
        transfers = prepare_transfers(ft + hit, out)
        message = "**Optimal Transfers for Current Team**"
        send_discord_message(message, [transfers, pitch])
    else:
        message = "**Current Team is optimized. Save your FT**"
        send_discord_message(message, [pitch])
