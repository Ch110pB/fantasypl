"""Helper functions for creating the FPL team images."""

import math
import operator
from functools import reduce
from itertools import starmap

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from fantasypl.config.constants import (
    KIT_IMAGE_HEIGHT,
    KIT_IMAGE_WIDTH,
    PITCH_IMAGE_HEIGHT,
    PITCH_IMAGE_WIDTH,
    RESOURCE_FOLDER,
    TRANSFER_BOX_HEIGHT,
    TRANSFER_BOX_WIDTH,
    TRANSFER_POINTER_IMAGE_SIZE,
)
from fantasypl.config.schemas import Season


def create_kit_with_textbox(
    player_code: int,
    player_name: str,
    season: Season,
    *,
    captain_player_code: int,
) -> Image.Image:
    """
    Build the kit image with player name.

    Parameters
    ----------
    player_code
        FPL player Code.
    player_name
        FPL player web name.
    season
        The season under process.
    captain_player_code
        FPL player code of the captain.

    Returns
    -------
        An image of the team kit with the player name.

    """
    font: ImageFont.FreeTypeFont = ImageFont.truetype(
        RESOURCE_FOLDER / "fonts/PremierLeagueW01-Bold.woff2",
        size=30,
    )
    shirt: str = f"photo_{player_code}.png"
    kit_image: Image.Image = (
        Image.open(
            RESOURCE_FOLDER / season.folder / "photos" / shirt,
        )
        .resize((KIT_IMAGE_WIDTH, KIT_IMAGE_WIDTH))
        .convert("RGBA")
    )
    name_box_image: Image.Image = Image.new(
        "RGBA",
        (KIT_IMAGE_WIDTH, KIT_IMAGE_HEIGHT - KIT_IMAGE_WIDTH),
        (255, 255, 255),
    )
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(name_box_image)
    draw.text(
        (KIT_IMAGE_WIDTH // 2, (KIT_IMAGE_HEIGHT - KIT_IMAGE_WIDTH) // 2),
        player_name,
        fill=(55, 0, 60),
        anchor="mm",
        font=font,
    )
    kit_image = Image.fromarray(
        np.vstack([np.asarray(kit_image), np.asarray(name_box_image)])
    )
    if player_code == captain_player_code:
        captain_image: Image.Image = (
            Image.open(
                RESOURCE_FOLDER / "img/captain.png",
            )
            .convert("RGBA")
            .resize((35, 35))
        )
        kit_image.paste(captain_image, (0, 0), captain_image)
    return kit_image


def paste_kits_on_pitch(
    pitch_image: Image.Image,
    vertical_position: int,
    list_elements: list[tuple[str, int, int]],
    captain: tuple[str, int],
    season: Season,
) -> Image.Image:
    """
    Paste the kits for a vertical position onto the pitch image.

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
        The pitch image with the elements pasted in.

    """
    i: int
    for i, (player_name, player_code, _) in enumerate(list_elements):
        kit_image: Image.Image
        kit_image = create_kit_with_textbox(
            player_code,
            player_name,
            season,
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


def create_transfer_packet(player_in: str, player_out: str) -> Image.Image:
    """
    Build the transfer packet for one transfer.

    Parameters
    ----------
    player_in
        Player name for the transferred in player.
    player_out
        Player name for the transferred out player.

    Returns
    -------
        A single transfer in-out image.

    """
    out_img: Image.Image = (
        Image.open(RESOURCE_FOLDER / "img" / "sub-off.png")
        .convert("RGBA")
        .resize((TRANSFER_POINTER_IMAGE_SIZE, TRANSFER_POINTER_IMAGE_SIZE))
    )
    in_img: Image.Image = (
        Image.open(RESOURCE_FOLDER / "img" / "sub-on.png")
        .convert("RGBA")
        .resize((TRANSFER_POINTER_IMAGE_SIZE, TRANSFER_POINTER_IMAGE_SIZE))
    )
    font: ImageFont.FreeTypeFont = ImageFont.truetype(
        RESOURCE_FOLDER / "fonts/PremierLeagueW01-Bold.woff2",
        size=30,
    )

    image: Image.Image = Image.new(
        "RGB",
        (TRANSFER_BOX_WIDTH, TRANSFER_BOX_HEIGHT),
        (255, 255, 255),
    )
    draw: ImageDraw.ImageDraw = ImageDraw.Draw(image)

    player_out_bbox: tuple[int, int, int, int] = draw.textbbox(
        (0, 0),
        player_out,
        font=font,
    )
    draw.text(
        (TRANSFER_BOX_WIDTH // 2 - 10, TRANSFER_BOX_HEIGHT // 2),
        player_out,
        fill=(55, 0, 60),
        anchor="ma",
        font=font,
    )
    image.paste(
        out_img,
        (
            TRANSFER_BOX_WIDTH // 2
            + (player_out_bbox[2] - player_out_bbox[0]) // 2
            - 7,
            TRANSFER_BOX_HEIGHT // 2
            + (player_out_bbox[3] - player_out_bbox[1]) // 2,
        ),
        out_img,
    )

    player_in_bbox: tuple[int, int, int, int] = draw.textbbox(
        (0, 0),
        player_in,
        font=font,
    )
    draw.text(
        (TRANSFER_BOX_WIDTH // 2 + 10, TRANSFER_BOX_HEIGHT // 2),
        player_in,
        fill=(55, 0, 60),
        anchor="md",
        font=font,
    )
    image.paste(
        in_img,
        (
            TRANSFER_BOX_WIDTH // 2
            - (player_in_bbox[2] - player_in_bbox[0]) // 2
            - TRANSFER_POINTER_IMAGE_SIZE
            + 7,
            TRANSFER_BOX_HEIGHT // 2
            - (player_in_bbox[3] - player_in_bbox[1]) // 2
            - TRANSFER_POINTER_IMAGE_SIZE,
        ),
        in_img,
    )
    return image


def get_image_grid(
    rows: int,
    cols: int,
    images: list[Image.Image],
) -> Image.Image:
    """
    Orient the transfer images for a given grid size.

    Parameters
    ----------
    rows
        Number of rows in transfer image grid.
    cols
        Number of columns in transfer image grid.
    images
        List of transfer images

    Returns
    -------
        Transfer image grid.

    """
    if len(images) < rows * cols:
        padding_image = Image.new('RGB', images[0].size, color=(255, 255, 255))
        images += [padding_image] * ((rows * cols) - len(images))
    image_arrays: list[npt.NDArray[np.int32]] = [
        np.asarray(img) for img in images
    ]
    image_rows: list[npt.NDArray[np.int32]] = [
        np.hstack(image_arrays[i : i + cols])
        for i in range(0, rows * cols, cols)
    ]
    return Image.fromarray(np.vstack(image_rows))


def prepare_transfers(
    transfers_in: list[str],
    transfers_out: list[str],
) -> Image.Image:
    """
    Build the transfers image depending on number of transfers.

    Parameters
    ----------
    transfers_in
        List of all inbound transfers.
    transfers_out
        List of all outbound transfers.

    Returns
    -------
        The image with all transfers.

    """
    images: list[Image.Image] = list(
        starmap(
            create_transfer_packet,
            zip(transfers_in, transfers_out, strict=True),
        ),
    )
    if len(images) <= 3:  # noqa: PLR2004
        return get_image_grid(len(images), 1, images)
    rows: int = math.ceil(math.sqrt(len(images)))
    cols: int = math.ceil(len(images) / rows)
    return get_image_grid(rows, cols, images)


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
    Build the final squad image with all the players on the pitch.

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
    pitch: Image.Image = (
        Image.open(
            RESOURCE_FOLDER / "img/pitch-default.png",
        )
        .convert("RGB")
        .crop((216, 0, 2016, PITCH_IMAGE_HEIGHT))
        .resize((
            PITCH_IMAGE_WIDTH,
            PITCH_IMAGE_HEIGHT,
        ))
    )
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

    i: int
    pos: list[tuple[str, int, int]]
    for i, pos in enumerate(eleven_players):
        pitch = paste_kits_on_pitch(pitch, i, pos, captain_player, season)
    return paste_kits_on_pitch(
        pitch,
        4,
        reduce(operator.add, sub_players),
        captain_player,
        season,
    )
