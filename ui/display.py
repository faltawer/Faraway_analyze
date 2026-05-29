from __future__ import annotations
import os
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "arial.ttf"


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except:
        return ImageFont.load_default()


def _get_text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple:
    """Compatible Pillow ancienne et nouvelle version."""
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def _stamp_score(img: Image.Image, score: int) -> Image.Image:
    """Ajoute le score en bas à gauche d'une carte."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    font = _load_font(28)
    text = f"{score} pts"
    tw, th = _get_text_size(draw, text, font)
    # Rectangle blanc + texte
    draw.rectangle([5, img.height - th - 14, tw + 15, img.height - 6], fill="white")
    draw.text((10, img.height - th - 10), text, fill="black", font=font)
    return img


def display_player_cards(
    played_card_ids: list[int],
    sanctuary_ids: list[int],
    card_folder: str,
    score_dict: dict[int, int] | None = None,
    output_path: str = "output.jpg",
) -> None:
    """
    Génère une image de toutes les cartes d'un joueur.

    Layout :
        Ligne 0       : sanctuaires (nb variable)
        Lignes 1-2    : 8 cartes jouées (4 par ligne)
        Coin haut droit : score total
    """

    def load(card_id: int) -> Image.Image:
        path = os.path.join(card_folder, f"card_{card_id}.jpg")
        return Image.open(path)

    if not played_card_ids:
        raise ValueError("played_card_ids ne peut pas être vide.")

    # Dimensions de référence
    sample = load(played_card_ids[0])
    cw, ch = sample.size
    pad = 10
    cols = 4

    num_rows = -(-len(played_card_ids) // cols)  # ceil division
    num_sanctuaries = len(sanctuary_ids)

    total_width = cols * cw + (cols + 1) * pad
    sanctuary_height = (ch + pad) if num_sanctuaries > 0 else 0
    total_height = sanctuary_height + num_rows * ch + (num_rows + 1) * pad

    canvas = Image.new("RGB", (total_width, total_height), "white")

    # --- Sanctuaires ---
    for idx, card_id in enumerate(sanctuary_ids):
        img = load(card_id)
        if score_dict is not None:
            img = _stamp_score(img, score_dict.get(card_id, 0))
        x = pad + idx * (cw + pad)
        y = pad
        canvas.paste(img, (x, y))
        draw = ImageDraw.Draw(canvas)
        text = f'{card_id}'
        font = _load_font(36)
        draw.text((x, y), text, fill="black", font=font)


    # --- Cartes jouées ---
    for i, card_id in enumerate(played_card_ids):
        img = load(card_id)
        if score_dict is not None:
            img = _stamp_score(img, score_dict.get(card_id, 0))
        row = i // cols
        col = i % cols
        x = pad + col * (cw + pad)
        y = sanctuary_height + pad + row * (ch + pad)
        canvas.paste(img, (x, y))

    # --- Score total en haut à droite ---
    if score_dict:
        total = sum(score_dict.values())
        draw = ImageDraw.Draw(canvas)
        font = _load_font(36)
        text = f"Total: {total} pts"
        tw, th = _get_text_size(draw, text, font)
        margin = 10
        x = canvas.width - tw - margin
        y = margin
        draw.rectangle([x - 5, y - 5, x + tw + 5, y + th + 5], fill="white")
        draw.text((x, y), text, fill="black", font=font)

    canvas.save(output_path)
    print(f"🖼️  Image sauvegardée : {output_path}")