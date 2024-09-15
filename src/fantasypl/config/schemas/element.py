"""Contains the superclass of Team and Player."""

from pydantic import BaseModel


class Element(BaseModel):
    """
    Superclass of Team and Player.

    Attributes
    ----------
        fbref_id: The ID in FBRef.
        fpl_code: The code in the FPL API.

    """

    fbref_id: str
    fpl_code: int

    def __eq__(self, other: object) -> bool:
        """
        Check equality of two Elements.

        Parameters
        ----------
        other
            The other object to compare.

        Returns
        -------
            Boolean value confirming equality.

        """
        if isinstance(other, Element):
            return self.fbref_id == other.fbref_id
        return False

    def __hash__(self) -> int:
        """
        Return the hash value of an Element.

        Returns
        -------
            The hash value of the Element.

        """
        return hash(self.fbref_id)
