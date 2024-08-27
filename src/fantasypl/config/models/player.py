"""Contains the Player class."""

from fantasypl.config.models.element_mixin import ElementMixin


class Player(ElementMixin):
    """
    The Player class.

    Attributes
    ----------
        fpl_full_name: Player full name in FPL API.
        fpl_web_name: Player web name in FPL API.
        fbref_name: Player full name in FBRef.

    """

    fpl_full_name: str
    fpl_web_name: str
    fbref_name: str
