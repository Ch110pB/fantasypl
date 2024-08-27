"""Contains the TeamGameweek class."""

import datetime
from typing import Literal

from pydantic import BaseModel

from fantasypl.config.models.team import Team


class TeamGameweek(BaseModel):
    """
    The TeamGameweek class.

    Attributes
    ----------
        team: The Team class object.
        season: Season full name in FBRef.
        date: Date of the match.
        venue: Home/Away for the team.
        formation: Team formation.
        formation_vs: Opponent formation.
        possession: Team possession.
        shots_on_target: Team shots on target.
        npxg: Team non-penalty expected goals.
        key_passes: Team key passes.
        pass_xa: Team total expected assists for completed passes.
        sca: Team shot creating actions.
        gca: Team goal creating actions.
        tackles_won_vs: Opponent tackles won.
        interceptions_vs: Opponent interceptions.
        blocks_vs: Opponent blocks.
        clearances_vs: Opponent clearances.
        npxg_vs: Opponent non-penalty expected goals.
        gk_saves_vs: Opponent goalkeeper saves.
        fouls_conceded: Team fouls conceded.
        yellow_cards: Team yellow cards.
        red_cards: Team red cards.
        fouls_won_vs: Opponent fouls won.
        yellow_cards_opp_vs: Opponent yellow cards against.
        red_cards_opp_vs: Opponent red cards against.
        pens_won: Team penalties won.
        pens_scored: Team penalties scored.
        pens_conceded_vs: Opponent penalties conceded against.

    """

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
