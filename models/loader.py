from __future__ import annotations
import pandas as pd
from .card import Card, CardType, Symbol, multiplicator_effect

# Dictionnaire de conversion string → Symbol
# Pour ajouter une extension : on ajoute juste une ligne ici
SYMBOL_MAP: dict[str, Symbol] = {
    "stone": Symbol.STONE,
    "chimera": Symbol.CHIMERA,
    "thistle": Symbol.THISTLE,
    "map": Symbol.MAP,
    "r": Symbol.RED,
    "b": Symbol.BLUE,
    "y": Symbol.YELLOW,
    "g": Symbol.GREEN,
    "night": Symbol.NIGHT,
    "day": Symbol.DAY,
}

CARD_TYPE_MAP: dict[str, CardType] = {
    "region": CardType.REGION,
    "sanctuary": CardType.SANCTUARY,
    "cards": CardType.REGION,
}

CONDITIONAL_MAP: dict[str, multiplicator_effect] = {
    "4colors": multiplicator_effect.FOUR_COLORS,
    "stone": multiplicator_effect.STONE,
    "chimera": multiplicator_effect.CHIMERA,
    "thistle": multiplicator_effect.THISTLE,
    "map": multiplicator_effect.MAP,
    "r": multiplicator_effect.RED,
    "b": multiplicator_effect.BLUE,
    "y": multiplicator_effect.YELLOW,
    "g": multiplicator_effect.GREEN,
    "night": multiplicator_effect.NIGHT,
    "day": multiplicator_effect.DAY,
}


def parse_conditional_list(raw) -> list[multiplicator_effect]:
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    return [CONDITIONAL_MAP[s.strip().lower()] for s in str(raw).split(",")]


def parse_symbol(raw: str) -> Symbol:
    """
    Convertit une string en Symbol.
    Lève une erreur claire si la valeur est inconnue.
    Utile pour détecter les fautes de frappe dans le fichier Excel.
    """
    key = raw.strip().lower()
    if key not in SYMBOL_MAP:
        raise ValueError(
            f"Symbole inconnu : '{raw}'. "
            f"Valeurs acceptées : {list(SYMBOL_MAP.keys())}"
        )
    return SYMBOL_MAP[key]


def parse_symbol_list(raw) -> list[Symbol]:
    """
    Convertit une cellule Excel (string ou NaN) en liste de Symbol.
    Exemple : "r, map, stone" → [Symbol.RED, Symbol.MAP, Symbol.STONE]
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    return [parse_symbol(s) for s in str(raw).split(",")]


def parse_card_type(raw: str) -> CardType:
    """Convertit une string en CardType."""
    key = str(raw).strip().lower()
    if key not in CARD_TYPE_MAP:
        raise ValueError(
            f"Type de carte inconnu : '{raw}'. "
            f"Valeurs acceptées : {list(CARD_TYPE_MAP.keys())}"
        )
    return CARD_TYPE_MAP[key]


def load_cards(card_file: str) -> tuple[list[Card], list[Card]]:
    """
    Charge les cartes depuis un fichier Excel.
    Retourne (cartes_region, cartes_sanctuaire).
    """
    df = pd.read_excel(card_file, engine="openpyxl")
    region_cards = []
    sanctuary_cards = []

    for _, row in df.iterrows():
        raw_color = row.get("color", "")
        card = Card(
            id=int(row["id"]),
            points=0 if pd.isna(row.get("points")) else int(row["points"]),
            color= "grey" if pd.isna(raw_color) else str(raw_color).strip(),
            card_type=parse_card_type(row.get("type", "region")),
            symbols=parse_symbol_list(row.get("symbols")),
            multiplicator_effect=parse_conditional_list(row.get("multiplicator")),
            activation_requirements=parse_symbol_list(row.get("activation_requirements")),
        )
        if card.card_type == CardType.SANCTUARY:
            sanctuary_cards.append(card)
        else:
            region_cards.append(card)

    return region_cards, sanctuary_cards
