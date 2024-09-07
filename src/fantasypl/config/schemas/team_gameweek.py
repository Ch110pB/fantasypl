"""Contains the TeamGameweek class."""

import datetime
from typing import Literal

from pydantic import BaseModel

from fantasypl.config.schemas.team import Team


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
        possession: Team possession.
        shots: Team total shots.
        shots_on_target: Team shots on target.
        average_shot_distance: Team shots average distance.
        npxg: Team non-penalty expected goals.
        npxg_vs: Opponent non-penalty expected goals.
        shots_on_target_vs: Opponent shots on target.
        pens_won: Team penalties won.
        pens_scored: Team penalties scored.
        passes_completed: Team completed passes.
        progressive_passes: Team progressive passes.
        key_passes: Team key passes.
        pass_xa: Team total expected assists for completed passes.
        passes_into_final_third: Team passes into the final third.
        progressive_carries: Team total progressive carries.
        sca: Team shot creating actions.
        gca: Team goal creating actions.
        sca_vs: Opponent shot creating actions.
        gca_vs: Opponent goal creating actions.
        tackles_won: Team tackles won.
        interceptions: Team interceptions.
        blocks: Team blocks.
        clearances: Team clearances.
        ball_recoveries: Team ball recoveries.
        aerials_won_pct: Team percentage aerial duels won.
        yellow_cards: Team yellow cards.
        red_cards: Team red cards.
        fouls_conceded: Team fouls conceded.
        fouls_won: Team fouls won.
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
    possession: int
    shots: int
    shots_on_target: int
    average_shot_distance: float
    npxg: float
    npxg_vs: float
    shots_on_target_vs: int
    pens_won: int
    pens_scored: int
    passes_completed: int
    progressive_passes: int
    key_passes: int
    pass_xa: float
    passes_into_final_third: int
    progressive_carries: int
    sca: int
    gca: int
    sca_vs: int
    gca_vs: int
    tackles_won: int
    interceptions: int
    blocks: int
    clearances: int
    ball_recoveries: int
    aerials_won_pct: float
    fouls_conceded: int
    fouls_won: int
    yellow_cards: int
    red_cards: int
    yellow_cards_vs: int
    red_cards_vs: int
    pens_conceded: int
    gk_saves: int
