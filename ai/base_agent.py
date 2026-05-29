from __future__ import annotations
from abc import ABC, abstractmethod
from models.card import Card
from models.game_state import GameState
from models.player import Player


class Agent(ABC):
    """
    Interface commune pour tous les agents (humain, random, IA).

    Règle : un Agent ne modifie JAMAIS le GameState directement.
    Il retourne des décisions, c'est engine/rules.py qui les applique.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def choose_card(self, player: Player, state: GameState) -> Card:
        """
        Choisit une carte à jouer depuis la main du joueur.
        Doit retourner une carte présente dans player.hand.
        """
        pass

    @abstractmethod
    def choose_sanctuary(self, player: Player, state: GameState) -> Card:
        """
        Choisit un sanctuaire parmi ceux tirés ce tour.
        Doit retourner une carte présente dans player.sanctuaries_drawn.
        """
        pass

    @abstractmethod
    def pick_from_center(self, player: Player, state: GameState) -> Card:
        """
        Choisit une carte parmi les cartes du centre.
        Doit retourner une carte présente dans state.middle_cards.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"