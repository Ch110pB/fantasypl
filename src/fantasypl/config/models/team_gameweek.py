import datetime
from typing import Literal

from pydantic import BaseModel

from fantasypl.config.models.team import Team


class TeamGameweek(BaseModel):
    team: Team
    season: str
    date: datetime.date
    venue: Literal["Home", "Away", "Neutral"]
    formation: str
    formation_vs: str
    possession: int
    shots_on_target: int
    npxg: float
    key_passes: int
    pass_xa: float
    sca: int
    gca: int
    tackles_won_vs: int
    interceptions_vs: int
    blocks_vs: int
    clearances_vs: int
    npxg_vs: float
    gk_saves_vs: int
    fouls_conceded: int
    yellow_cards: int
    red_cards: int
    fouls_won_vs: int
    yellow_cards_opp_vs: int
    red_cards_opp_vs: int
    pens_won: int
    pens_scored: int
    pens_conceded_vs: int
