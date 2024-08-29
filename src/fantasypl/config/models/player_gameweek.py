"""Contains the PlayerGameweek class."""

import datetime
from typing import Literal

from pydantic import BaseModel

from fantasypl.config.models.player import Player
from fantasypl.config.models.team import Team


class PlayerGameWeek(BaseModel):
    """
    The PlayerGameweek class.

    Attributes
    ----------
        player: The player class object.
        team: The Team class object.
        season: Season full name in FBRef.
        date: Date of the match.
        venue: Home/Away for the team.
        short_position: Player short position (GK/DF/MF/FW).
        minutes: Player minutes played.
        starts: True if player started the match, False otherwise.
        shots_on_target: Player shots on target.
        npxg: Player non-penalty expected goals.
        key_passes: Player key passes.
        pass_xa: Player total expected assists for completed passes.
        xa: Player expected assists.
        yellow_cards: Player yellow cards.
        red_cards: Player red cards.
        sca: Player shot creating actions.
        gca: Player goal creating actions.
        pens_taken: Player penalties taken.
        pens_scored: Player penalties scored.
        progressive_passes: Player progressive passes.
        progressive_carries: Player progressive carries.
        tackles_won: Player tackles won.
        blocks: Player blocks.
        interceptions: Player interceptions.
        clearances: Player clearances.
        fouls: Player fouls.
        gk_saves: Goalkeeper saves.
        gk_psxg: Goalkeeper post-shot expected goals faced.

    """

    player: Player
    team: Team
    season: str
    date: datetime.date
    venue: Literal["Home", "Away", "Neutral"]
    short_position: Literal["GK", "DF", "MF", "FW", None]
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
