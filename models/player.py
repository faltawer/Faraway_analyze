from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from .card import Card


@dataclass
class Player:
    name: str
    hand: List[Card] = field(default_factory=list)
    played_cards: List[Card] = field(default_factory=list)
    sanctuaries: List[Card] = field(default_factory=list)
    sanctuaries_drawn: List[Card] = field(default_factory=list)
    current_card: Optional[Card] = None
    drawn_card: Optional[Card] = None

    def __repr__(self):
        return (f"Player({self.name} | "
                f"main:{len(self.hand)} | "
                f"posées:{self.played_cards} | "
                f"sanctuaires:{self.sanctuaries})"
                f"piochée:{self.drawn_card})")
