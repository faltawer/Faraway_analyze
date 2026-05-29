from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from copy import deepcopy
from .card import Card
from .player import Player

@dataclass
class GameState:
    players: List[Player]
    deck: List[Card]
    middle_cards: List[Card]
    sanctuary_deck: List[Card]
    current_round: int = 1

    def clone(self) -> "GameState":
        """Copie profonde — l'IA peut simuler sans affecter la vraie partie."""
        return deepcopy(self)

    @property
    def is_finished(self) -> bool:
        return self.current_round > 8

    def __repr__(self):
        return (f"GameState(round {self.current_round}/8 | "
                f"deck:{len(self.deck)} | "
                f"centre:{len(self.middle_cards)})")