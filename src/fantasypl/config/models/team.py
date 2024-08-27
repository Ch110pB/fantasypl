"""Contains the Team class."""

from fantasypl.config.models.element_mixin import ElementMixin


class Team(ElementMixin):
    """
    The Team class.

    Attributes
    ----------
        fpl_name: Team name in FPL API.
        fbref_name: Team name in FBRef.
        short_name: Team short name in FPL API.

    """

    fpl_name: str
    fbref_name: str
    short_name: str
