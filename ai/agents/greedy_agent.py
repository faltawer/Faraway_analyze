from __future__ import annotations

import random

from models.card import Card
from models.game_state import GameState
from models.player import Player
from engine.scorer import score_player
from ai.base_agent import Agent


class GreedyAgent(Agent):
    """
    Choisit la carte qui maximise le score immédiat simulé.

    Stratégie : pour chaque carte en main, on simule qu'on la joue
    et on calcule le score actuel. On garde la meilleure.

    Limite : ne voit pas plus loin qu'un tour (pas d'anticipation).
    C'est l'étape 2 — l'étape 3 ajoutera le lookahead.
    """

    def choose_card(self, player: Player, state: GameState) -> Card:
        best_card = None
        best_score = -1

        for card in player.hand:
            score = self._simulate_card(card, player)
            if score > best_score:
                best_score = score
                best_card = card

        return best_card

    def pick_from_center(self, player, state):
        return max(state.middle_cards,
                   key=lambda c: self._simulate_card(c, player))

    def choose_sanctuary(self, player: Player, state: GameState) -> Card:
        """
        Choisit le sanctuaire qui maximise le score immédiat.
        """
        best_sanctuary = None
        best_score = -1

        for sanctuary in player.sanctuaries_drawn:
            score = self._simulate_sanctuary(sanctuary, player)
            if score > best_score:
                best_score = score
                best_sanctuary = sanctuary
        player.sanctuaries.append(best_sanctuary)  # Ajoute le sanctuaire choisi
        player.sanctuaries_drawn.clear()


    def _simulate_card(self, card: Card, player: Player) -> int:
        """
        Simule la pose d'une carte et retourne le score obtenu.
        Ne modifie pas le vrai joueur — travaille sur une copie.
        """
        from copy import deepcopy
        simulated = deepcopy(player)
        simulated.played_cards.append(card)
        score, _ = score_player(simulated)
        return score

    def _simulate_sanctuary(self, sanctuary: Card, player: Player) -> int:
        """
        Simule l'ajout d'un sanctuaire et retourne le score obtenu.
        """
        from copy import deepcopy
        simulated = deepcopy(player)
        simulated.sanctuaries.append(sanctuary)
        score, _ = score_player(simulated)
        return score