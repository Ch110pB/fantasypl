"""Contains the TeamGameweek class."""

import datetime
from typing import Literal

from pydantic import BaseModel

from fantasypl.config.schemas import Team


class TeamGameweek(BaseModel):
    """
    The TeamGameweek class.

    Attributes
    ----------
        team: The Team object.
        opponent: The Team object for opponent team.
        season: Season full name in FBRef.
        date: Date of the match.
        venue: Home/Away match.
        formation: Team formation.
        formation_vs: Opponent formation.
        possession: Team possession.
        shots_on_target: Team shots on target.
        npxg: Team non-penalty expected goals.
        pens_won: Team penalties won.
        pens_scored: Team penalties scored.
        shots_on_target_vs: Opponent shots on target.
        npxg_vs: Opponent non-penalty expected goals.
        key_passes: Team key passes.
        pass_xa: Team total expected assists for completed passes.
        sca: Team shot creating actions.
        gca: Team goal creating actions.
        tackles_won: Team tackles won.
        interceptions: Team interceptions.
        blocks: Team blocks.
        clearances: Team clearances.
        fouls_conceded: Team fouls conceded.
        fouls_won: Team fouls won.
        yellow_cards: Team yellow cards.
        red_cards: Team red cards.
        pens_conceded: Team penalties conceded.
        yellow_cards_vs: Opponent yellow cards.
        red_cards_vs: Opponent red cards.
        gk_saves: Team goalkeeper saves.

    """

    team: Team
    opponent: Team
    season: str
    date: datetime.date
    venue: Literal["Home", "Away"]
    formation: str
    formation_vs: str
    possession: int
    shots_on_target: int
    npxg: float
    pens_won: int
    pens_scored: int
    shots_on_target_vs: int
    npxg_vs: float
    key_passes: int
    pass_xa: float
    sca: int
    gca: int
    tackles_won: int
    interceptions: int
    blocks: int
    clearances: int
    fouls_conceded: int
    fouls_won: int
    yellow_cards: int
    red_cards: int
    yellow_cards_vs: int
    red_cards_vs: int
    pens_conceded: int
    gk_saves: int
