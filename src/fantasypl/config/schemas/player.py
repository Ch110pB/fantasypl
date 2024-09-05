"""Contains the Player class."""

from fantasypl.config.schemas.element import Element


class Player(Element):
    """
    The Player class.

    Attributes
    ----------
        fpl_full_name: Player full name in the FPL API.
        fpl_web_name: Player web name in the FPL API.
        fbref_name: Player full name in FBRef.

    """

    fpl_full_name: str
    fpl_web_name: str
    fbref_name: str
