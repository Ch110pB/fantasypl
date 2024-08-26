from fantasypl.config.models.element_mixin import ElementMixin


class Player(ElementMixin):
    fpl_full_name: str
    fpl_web_name: str
    fbref_name: str
