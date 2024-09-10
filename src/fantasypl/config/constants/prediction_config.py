"""Variables required during prediction time."""

FPL_POSITION_ID_DICT: dict[int, str] = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POINTS_GOALS: dict[str, int] = {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
POINTS_CS: dict[str, int] = {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
POINTS_SAVES: dict[str, int] = {"GKP": 1, "DEF": 0, "MID": 0, "FWD": 0}
POINTS_GOALS_CONCEDED: dict[str, int] = {
    "GKP": -1,
    "DEF": -1,
    "MID": 0,
    "FWD": 0,
}

TOTAL_LINEUP_COUNT = 11
MIN_GKP_COUNT: int = 1
MAX_GKP_COUNT: int = 1
TOTAL_GKP_COUNT: int = 2
MAX_DEF_COUNT: int = 5
MIN_DEF_COUNT: int = 3
TOTAL_DEF_COUNT: int = 5
MAX_MID_COUNT: int = 5
MIN_MID_COUNT: int = 3
TOTAL_MID_COUNT: int = 5
MAX_FWD_COUNT: int = 3
MIN_FWD_COUNT: int = 1
TOTAL_FWD_COUNT: int = 3
MAX_SAME_CLUB_COUNT: int = 3


TEAM_PREDICTION_SCALING_FACTORS: dict[str, dict[str, float]] = {
    "xgoals": {"mean": 1.44, "std": 0.64},
    "xyc": {"mean": 2.17, "std": 1.17},
    "xpens": {"mean": 0.12, "std": 0.28},
}
"""
The scaling factor for the team-level model predictions
to better match the true spread of the stats.
"""

MINUTES_STANDARD_DEVIATION: float = 31.72
"""
The standard deviation of predicted playing minutes
is 31.72.
"""

BENCH_WEIGHTS_ARRAY: list[float] = [0.05, 0.495, 0.171, 0.021]
"""
From a set of about 1500 managers from top FPL leagues in
GW 1-3 in 2024-25 season, this ratio was calculated from
taking the GW 1 subs, and seeing how many times they have
featured in the following two weeks, plus any auto-subs
that have happened.
"""

WEIGHTS_DECAYS_BASE: list[float] = [0.865, 0.585]
"""
0.48 is the linear regression slope between one week's
top scorers' next week score and the max score of next week
for 2023-24. Then adding total information kept by each week,
normalizing to unit sum and then dividing by first week, we
find the decay score.
"""

TRANSFER_HIT_PENALTY_PERCENTILE: float = 77.03
"""
FPL transfer hit penalty is 4, which computes to 77.03
percentile of all FPL gameweek scores from the last 3
seasons for players with minutes > 0.
"""

TRANSFER_GAIN_MINIMUM: float = 0.0404 * (1 + sum(WEIGHTS_DECAYS_BASE))
"""
Standard deviation for the current model is 0.32, which equates to
a standard error of 0.0146 with predicted gameweek points for
gameweek 3. Using z-score of 95% confidence interval we get 0.0404
as the minimum difference between two predictions for their
difference to be statistically significant. Then multiplying it by
the weighting scheme gives us final minimum difference.
"""
