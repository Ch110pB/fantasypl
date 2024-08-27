"""Contains the superclass of Team and Player."""

from pydantic import BaseModel


class ElementMixin(BaseModel):
    """
    Superclass of Team and Player.

    Attributes
    ----------
        fbref_id: The ID in FBRef.
        fpl_code: The code in FPL API.

    """

    fbref_id: str
    fpl_code: int

    def __eq__(self, other: object) -> bool:
        """

        Args:
        ----
            other: The other ElementMixin object to compare equality.

        Returns:
        -------
            A boolean value confirming equality between two ElementMixin objects.

        """
        if isinstance(other, ElementMixin):
            return self.fbref_id == other.fbref_id
        return False

    def __hash__(self) -> int:
        """

        Returns
        -------
            Hash value of a ElementMixin object.

        """
        return hash(self.fbref_id)
