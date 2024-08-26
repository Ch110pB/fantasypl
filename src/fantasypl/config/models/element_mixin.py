from pydantic import BaseModel


class ElementMixin(BaseModel):
    fbref_id: str
    fpl_code: int

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ElementMixin):
            return self.fbref_id == other.fbref_id
        return False

    def __hash__(self) -> int:
        return hash(self.fbref_id)
