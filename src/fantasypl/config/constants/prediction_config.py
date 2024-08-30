"""Dictionaries required during prediction time."""

FPL_POSITION_ID_DICT: dict[int, str] = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POINTS_GOALS: dict[str, int] = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
POINTS_CS: dict[str, int] = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
POINTS_SAVES: dict[str, int] = {"GKP": 1, "DEF": 0, "MID": 0, "FWD": 0}
POINTS_GOALS_CONCEDED: dict[str, int] = {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0}
