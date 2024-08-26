from fantasypl.config.models.element_mixin import ElementMixin


class Team(ElementMixin):
    fpl_name: str
    fbref_name: str
    short_name: str
