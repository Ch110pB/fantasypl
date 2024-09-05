"""Contains the Team class."""

from fantasypl.config.schemas.element import Element


class Team(Element):
    """
    The Team class.

    Attributes
    ----------
        fpl_name: Team name in the FPL API.
        fbref_name: Team name in FBRef.
        short_name: Team short name in the FPL API.

    """

    fpl_name: str
    fbref_name: str
    short_name: str
