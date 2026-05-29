from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class CardType(Enum):
    REGION = "region"
    SANCTUARY = "sanctuary"

class Symbol(Enum):
    STONE   = "stone"
    CHIMERA = "chimera"
    THISTLE = "thistle"
    MAP     = "map"
    RED     = "r"
    BLUE    = "b"
    YELLOW  = "y"
    GREEN   = "g"
    NIGHT   = "night"
    DAY     = "day"

class multiplicator_effect(Enum):
    """Effets conditionnels spéciaux — distincts des symboles ordinaires."""
    FOUR_COLORS = "4colors"   # 1 pt par set complet r+b+y+g
    STONE       = "stone"     # X pts par symbole stone visible
    CHIMERA     = "chimera"
    THISTLE     = "thistle"
    MAP         = "map"
    RED         = "r"
    BLUE        = "b"
    YELLOW      = "y"
    GREEN       = "g"
    NIGHT       = "night"
    DAY         = "day"

@dataclass
class Card:
    id: int
    points: int
    color: str
    card_type: CardType
    symbols: List[Symbol]               = field(default_factory=list)
    multiplicator_effect: List[Symbol]    = field(default_factory=list)
    activation_requirements: List[Symbol] = field(default_factory=list)

    def __repr__(self):
        return f"<Card {self.id} | {self.card_type.value} | {self.points}pts>"