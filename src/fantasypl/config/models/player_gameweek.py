import datetime
from typing import Literal

from pydantic import BaseModel

from fantasypl.config.models.player import Player
from fantasypl.config.models.team import Team


class PlayerGameWeek(BaseModel):
    player: Player
    team: Team
    season: str
    date: datetime.date
    venue: Literal["Home", "Away", "Neutral"]
    short_position: Literal["GK", "DF", "MF", "FW"]
    minutes: int
    starts: bool
    shots_on_target: int
    npxg: float
    key_passes: int
    pass_xa: float
    xa: float
    yellow_cards: int
    red_cards: int
    sca: int
    gca: int
    pens_taken: int
    pens_scored: int
    progressive_passes: int
    progressive_carries: int
    tackles_won: int
    blocks: int
    interceptions: int
    clearances: int
    fouls: int
    gk_saves: int
    gk_psxg: float
