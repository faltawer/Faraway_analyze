from __future__ import annotations
import random
from models.card import Card
from models.game_state import GameState
from models.player import Player
from ai.base_agent import Agent


class RandomAgent(Agent):
    """
    Choisit toujours au hasard.
    Sert de baseline : toute IA doit faire mieux que ça.
    """

    def choose_card(self, player: Player, state: GameState) -> Card:
        return random.choice(player.hand)

    def choose_sanctuary(self, player: Player, state: GameState) -> Card:
        player.sanctuaries.append(random.choice(player.sanctuaries_drawn))  # Ajoute le sanctuaire choisi
        player.sanctuaries_drawn.clear()

    def pick_from_center(self, player: Player, state: GameState) -> Card:
        """
        Pioche une carte aléatoire parmi les cartes du centre.
        """
        if not state.middle_cards:
            raise ValueError("Aucune carte disponible au centre.")
        return random.choice(state.middle_cards)