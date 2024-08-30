"""
Mapping dictionaries.

Contains:
- FBREF_LEAGUE_OPTA_STRENGTH_DICT:
        Dictionary containing league strengths of last season (Opta Power Rankings).
- FBREF_POSITION_MAPPING:
        Dictionary containing FBRef position to short_position mapping.
"""

FBREF_LEAGUE_OPTA_STRENGTH_DICT: dict[str, float] = {
    "eng ENG_1. Premier League": 86.5,
    "es ESP_1. La Liga": 84.9,
    "ch SUI_1. Super Lg": 77.35,
    "it ITA_1. Serie A": 86.3,
    "au AUS_1. A-League": 70.0,
    "eng ENG_2. Championship": 77.5,
    "de GER_1. Bundesliga": 84.7,
    "eng ENG_4. League Two": 60.75,
    "rs SRB_1. SuperLiga": 65.7,
    "eng ENG_3. League One": 69.7,
    "fr FRA_1. Ligue 1": 85.7,
    "tr TUR_1. Süper Lig": 76.0,
    "be BEL_1. Pro League A": 78.75,
    "se SWE_1. Allsvenskan": 74.5,
    "ar ARG_1. Liga Argentina": 76.9,
    "dk DEN_1. Danish Superliga": 77.25,
    "nl NED_1. Eredivisie": 74.9,
    "ro ROU_1. Liga I": 73.7,
    "sct SCO_1. Premiership": 71.45,
    "br BRA_1. Série A": 80.45,
    "us USA_1. MLS": 78.4,
    "pt POR_1. Primeira Liga": 77.25,
    "de GER_2. 2. Bundesliga": 76.45,
    "gr GRE_1. Super League": 70.0,
    "kr KOR_1. K League": 75.4,
    "nl NED_2. Eerste Divisie": 64.85,
    "at AUT_1. Bundesliga": 75.5,
    "py PAR_1. Primera Div": 69.35,
}

FBREF_POSITION_MAPPING: dict[str, str] = {
    "GK": "GK",
    "DF": "DF",
    "MF": "MF",
    "FW": "FW",
    "FB": "DF",
    "LB": "DF",
    "RB": "DF",
    "CB": "DF",
    "WB": "DF",
    "DM": "MF",
    "CM": "MF",
    "LM": "MF",
    "RM": "MF",
    "WM": "MF",
    "AM": "MF",
    "LW": "FW",
    "RW": "FW",
}
