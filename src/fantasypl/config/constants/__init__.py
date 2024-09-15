"""Exposes all the inner constants for a folder level import."""

from .folder_config import (
    DATA_FOLDER_FBREF,
    DATA_FOLDER_FPL,
    DATA_FOLDER_REF,
    MODEL_FOLDER,
    RESOURCE_FOLDER,
)
from .image_config import (
    BENCH_VERTICAL_POSITION,
    KIT_IMAGE_HEIGHT,
    KIT_IMAGE_WIDTH,
    PITCH_IMAGE_HEIGHT,
    PITCH_IMAGE_WIDTH,
    TRANSFER_BOX_HEIGHT,
    TRANSFER_BOX_WIDTH,
    TRANSFER_POINTER_IMAGE_SIZE,
)
from .mapping_config import (
    FBREF_LEAGUE_OPTA_STRENGTH_DICT,
    FBREF_POSITION_MAPPING,
)
from .modeling_config import (
    METRIC,
    MODELS,
    SEED,
    SPLITS_CV,
    TASK,
    TIME_TRAINING_PLAYER,
    TIME_TRAINING_TEAM,
)
from .prediction_config import (
    BENCH_WEIGHTS_ARRAY,
    FPL_POSITION_ID_DICT,
    MAX_DEF_COUNT,
    MAX_FWD_COUNT,
    MAX_GKP_COUNT,
    MAX_MID_COUNT,
    MAX_SAME_CLUB_COUNT,
    MIN_DEF_COUNT,
    MIN_FWD_COUNT,
    MIN_GKP_COUNT,
    MIN_MID_COUNT,
    MINUTES_STANDARD_DEVIATION,
    POINTS_CS,
    POINTS_GOALS,
    POINTS_GOALS_CONCEDED,
    POINTS_SAVES,
    TEAM_PREDICTION_SCALING_FACTORS,
    TOTAL_DEF_COUNT,
    TOTAL_FWD_COUNT,
    TOTAL_GKP_COUNT,
    TOTAL_LINEUP_COUNT,
    TOTAL_MID_COUNT,
    TRANSFER_GAIN_MINIMUM,
    TRANSFER_HIT_PENALTY_PERCENTILE,
    WEIGHTS_DECAYS_BASE,
)
from .web_config import (
    FBREF_BASE_URL,
    FPL_BADGES_URL,
    FPL_BOOTSTRAP_URL,
    FPL_FIXTURES_URL,
    FPL_SHIRTS_URL,
    FPL_TEAM_URL,
)


__all__ = [
    "BENCH_VERTICAL_POSITION",
    "BENCH_WEIGHTS_ARRAY",
    "DATA_FOLDER_FBREF",
    "DATA_FOLDER_FPL",
    "DATA_FOLDER_REF",
    "FBREF_BASE_URL",
    "FBREF_LEAGUE_OPTA_STRENGTH_DICT",
    "FBREF_POSITION_MAPPING",
    "FPL_BADGES_URL",
    "FPL_BOOTSTRAP_URL",
    "FPL_FIXTURES_URL",
    "FPL_POSITION_ID_DICT",
    "FPL_SHIRTS_URL",
    "FPL_TEAM_URL",
    "KIT_IMAGE_HEIGHT",
    "KIT_IMAGE_WIDTH",
    "MAX_DEF_COUNT",
    "MAX_FWD_COUNT",
    "MAX_GKP_COUNT",
    "MAX_MID_COUNT",
    "MAX_SAME_CLUB_COUNT",
    "METRIC",
    "MINUTES_STANDARD_DEVIATION",
    "MIN_DEF_COUNT",
    "MIN_FWD_COUNT",
    "MIN_GKP_COUNT",
    "MIN_MID_COUNT",
    "MODELS",
    "MODEL_FOLDER",
    "PITCH_IMAGE_HEIGHT",
    "PITCH_IMAGE_WIDTH",
    "POINTS_CS",
    "POINTS_GOALS",
    "POINTS_GOALS_CONCEDED",
    "POINTS_SAVES",
    "RESOURCE_FOLDER",
    "SEED",
    "SPLITS_CV",
    "TASK",
    "TEAM_PREDICTION_SCALING_FACTORS",
    "TIME_TRAINING_PLAYER",
    "TIME_TRAINING_TEAM",
    "TOTAL_DEF_COUNT",
    "TOTAL_FWD_COUNT",
    "TOTAL_GKP_COUNT",
    "TOTAL_LINEUP_COUNT",
    "TOTAL_MID_COUNT",
    "TRANSFER_BOX_HEIGHT",
    "TRANSFER_BOX_WIDTH",
    "TRANSFER_GAIN_MINIMUM",
    "TRANSFER_HIT_PENALTY_PERCENTILE",
    "TRANSFER_POINTER_IMAGE_SIZE",
    "WEIGHTS_DECAYS_BASE",
]
