"""Helper functions for creating the FPL team image."""

import operator
from functools import reduce

from PIL import Image, ImageDraw, ImageFont

from fantasypl.config.constants import (
    KIT_IMAGE_HEIGHT,
    KIT_IMAGE_WIDTH,
    PITCH_IMAGE_HEIGHT,
    PITCH_IMAGE_WIDTH,
    RESOURCE_FOLDER,
)
from fantasypl.config.schemas import Season


def create_kit_with_textbox(  # noqa: PLR0913
    team_code: int,
    player_code: int,
    player_name: str,
    season: Season,
    *,
    is_gk: bool,
    captain_player_code: int,
) -> Image.Image:
    """

    Parameters
    ----------
    team_code
        FPL Team Code.
    player_code
        FPL Player Code.
    player_name
        FPL Player web name.
    season
        The season under process.
    is_gk
        Boolean if goalkeeper kits are to be used.
    captain_player_code
        The captain.

    Returns
    -------
        A image of the team kit with the player name.

    """
    font: ImageFont.FreeTypeFont = ImageFont.truetype(
        RESOURCE_FOLDER / "fonts/PremierLeagueW01-Bold.woff2", size=30
    )
    shirt_name: str = (
        f"shirt_{team_code}_gk.png" if is_gk else f"shirt_{team_code}.png"
    )
    kit_image: Image.Image = Image.open(
        RESOURCE_FOLDER / season.folder / "shirts" / shirt_name
    ).convert("RGBA")

    draw: ImageDraw.ImageDraw = ImageDraw.Draw(kit_image)
    draw.rounded_rectangle(
        xy=[(0, KIT_IMAGE_WIDTH), (KIT_IMAGE_WIDTH, KIT_IMAGE_HEIGHT)],
        radius=10,
        fill=(255, 255, 255),
        outline=(55, 0, 60),
    )
    draw.text(
        (
            KIT_IMAGE_WIDTH // 2,
            KIT_IMAGE_WIDTH + (KIT_IMAGE_HEIGHT - KIT_IMAGE_WIDTH) // 2,
        ),
        player_name,
        fill=(55, 0, 60),
        anchor="mm",
        font=font,
    )
    if player_code == captain_player_code:
        captain_image: Image.Image = Image.open(
            RESOURCE_FOLDER / "img/captain.png"
        ).convert("RGBA")
        captain_image = captain_image.resize((35, 35))
        kit_image.paste(captain_image)
    return kit_image


def paste_kits_on_pitch(
    pitch_image: Image.Image,
    vertical_position: int,
    list_elements: list[tuple[str, int, int]],
    captain: tuple[str, int],
    season: Season,
) -> Image.Image:
    """

    Parameters
    ----------
    pitch_image
        The PIL image of the pitch.
    vertical_position
        Vertical position of the elements in the pitch.
    list_elements
        The list of elements
    captain
        The captain.
    season
        The season under process.

    Returns
    -------
        The pitch image with the elements.

    """
    i: int
    element: tuple[str, int, int]
    for i, element in enumerate(list_elements):
        player_name: str
        team_code: int
        player_name, player_code, team_code = element
        kit_image: Image.Image
        if vertical_position in {1, 2, 3}:
            kit_image = create_kit_with_textbox(
                team_code,
                player_code,
                player_name,
                season,
                is_gk=False,
                captain_player_code=captain[1],
            )
        elif vertical_position == 0:
            kit_image = create_kit_with_textbox(
                team_code,
                player_code,
                player_name,
                season,
                is_gk=True,
                captain_player_code=captain[1],
            )
        elif i > 0:
            kit_image = create_kit_with_textbox(
                team_code,
                player_code,
                player_name,
                season,
                is_gk=False,
                captain_player_code=captain[1],
            )
        else:
            kit_image = create_kit_with_textbox(
                team_code,
                player_code,
                player_name,
                season,
                is_gk=True,
                captain_player_code=captain[1],
            )
        pitch_image.paste(
            kit_image,
            (
                (2 * i + 1)
                * (PITCH_IMAGE_WIDTH - KIT_IMAGE_WIDTH)
                // (len(list_elements) * 2),
                vertical_position
                * (
                    KIT_IMAGE_HEIGHT
                    + (PITCH_IMAGE_HEIGHT - 5 * KIT_IMAGE_HEIGHT) // 5
                ),
            ),
            kit_image,
        )
    return pitch_image


def prepare_pitch(
    eleven_players: tuple[
        list[tuple[str, int, int]],
        list[tuple[str, int, int]],
        list[tuple[str, int, int]],
        list[tuple[str, int, int]],
    ],
    sub_players: tuple[
        list[tuple[str, int, int]],
        list[tuple[str, int, int]],
        list[tuple[str, int, int]],
        list[tuple[str, int, int]],
    ],
    captain_player: tuple[str, int],
    season: Season,
) -> Image.Image:
    """

    Parameters
    ----------
    eleven_players
        The players in lineup.
    sub_players
        The players in bench.
    captain_player
        The captain.
    season
        The season under process.

    Returns
    -------
        The pitch with all players.

    """
    pitch: Image.Image = Image.open(
        RESOURCE_FOLDER / "img/pitch-default.png"
    ).convert("RGB")
    pitch = pitch.resize((PITCH_IMAGE_WIDTH, PITCH_IMAGE_HEIGHT))
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(pitch, mode="RGBA")
    draw.rectangle(
        xy=[
            (
                0,
                PITCH_IMAGE_HEIGHT
                - KIT_IMAGE_HEIGHT
                - (PITCH_IMAGE_HEIGHT - 5 * KIT_IMAGE_HEIGHT) // 3.5,
            ),
            (PITCH_IMAGE_WIDTH, PITCH_IMAGE_HEIGHT),
        ],
        fill=(46, 139, 87, int(255 * 0.8)),
    )

    for i, pos in enumerate(eleven_players):
        pitch = paste_kits_on_pitch(pitch, i, pos, captain_player, season)
    return paste_kits_on_pitch(
        pitch, 4, reduce(operator.add, sub_players), captain_player, season
    )
