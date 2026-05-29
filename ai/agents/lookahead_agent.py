from __future__ import annotations
import random
from copy import deepcopy

from models.card import Card
from models.player import Player
from models.game_state import GameState
from engine.scorer import score_player
from engine.rules import (
    get_play_order,
    draw_sanctuaries,
    discard_excess_sanctuaries,
    replenish_center,
)
from ai.base_agent import Agent


class LookaheadAgent(Agent):
    """
    IA par arbre de recherche en profondeur.

    À chaque tour, pour chaque carte jouable :
        - Clone le GameState
        - Simule `depth` tours en avant (adversaires = random)
        - Évalue le score final
        - Choisit la carte qui maximise ce score

    `n_simulations` : nombre de simulations par carte candidate
    (les adversaires étant random, on moyenne sur plusieurs tirages)
    """

    def __init__(self, name: str, depth: int = 3, n_simulations: int = 20):
        super().__init__(name)
        self.depth = depth
        self.n_simulations = n_simulations

    def choose_card(self, player: Player, state: GameState) -> Card:
        best_card = None
        best_score = -1

        for card in player.hand:
            avg_score = self._evaluate_card(card, player, state)
            if avg_score > best_score:
                best_score = avg_score
                best_card = card

        return best_card

    def choose_sanctuary(self, player: Player, state: GameState) -> Card:
        """Choisit un sanctuaire en utilisant Q-Learning et met à jour l'état du joueur."""
        if not player.sanctuaries_drawn:
            return None

        # Choix du sanctuaire (Q-Learning ou aléatoire)
        if random.random() < self.epsilon:
            chosen = random.choice(player.sanctuaries_drawn)
        else:
            best = None
            best_q = -1
            state_key = self._state_key(player, state)
            for sanctuary in player.sanctuaries_drawn:
                q = self.Q[state_key].get(sanctuary.id, 0)  # Utilise .get() pour éviter KeyError
                if q > best_q:
                    best_q = q
                    best = sanctuary
            chosen = best if best else random.choice(player.sanctuaries_drawn)

        # Mise à jour de l'état du joueur (comme dans la version aléatoire)
        player.sanctuaries.append(chosen)
        player.sanctuaries_drawn.clear()

        return chosen

    def _evaluate_card(
        self, card: Card, player: Player, state: GameState
    ) -> float:
        """
        Moyenne du score final sur `n_simulations` parties simulées,
        en supposant qu'on joue `card` maintenant.
        """
        total = 0
        for _ in range(self.n_simulations):
            total += self._simulate(card, player, state)
        return total / self.n_simulations

    def _simulate(
        self, card: Card, player: Player, state: GameState
    ) -> int:
        """
        Simule une partie depuis l'état actuel :
        1. On joue `card`
        2. Les `depth` tours suivants sont simulés avec des agents random
        3. On retourne le score final du joueur
        """
        # Clone complet pour ne pas modifier l'état réel
        sim_state = state.clone()

        # Trouver le joueur simulé correspondant
        sim_player = next(
            p for p in sim_state.players if p.name == player.name
        )

        # Jouer la carte choisie
        sim_player.hand.remove(card)
        sim_player.played_cards.append(card)
        sim_player.current_card = card

        # Simuler les adversaires ce tour (random)
        for p in sim_state.players:
            if p.name != player.name and p.hand:
                chosen = random.choice(p.hand)
                p.hand.remove(chosen)
                p.played_cards.append(chosen)
                p.current_card = chosen

        # Appliquer les règles du tour
        self._apply_round_rules(sim_state)

        # Simuler les tours suivants avec tous les joueurs en random
        remaining = min(self.depth, 8 - sim_state.current_round)
        for _ in range(remaining):
            if sim_state.is_finished:
                break
            self._play_random_round(sim_state)

        # Score final du joueur simulé
        score, _ = score_player(sim_player)
        return score

    def _apply_round_rules(self, state: GameState) -> None:
        """Applique sanctuaires + pioche + renouvellement centre."""
        order = get_play_order(state.players)

        if state.current_round > 1:
            for p in order:
                draw_sanctuaries(p, state)
            for p in order:
                if p.sanctuaries_drawn:
                    # Choix greedy simple pour la simulation
                    best = max(
                        p.sanctuaries_drawn,
                        key=lambda s: score_player(
                            _player_with_sanctuary(p, s)
                        )[0]
                    )
                    p.sanctuaries.append(best)
                    p.sanctuaries_drawn.clear()
                discard_excess_sanctuaries(p, state)

        if state.current_round < 8:
            for p in order:
                if state.middle_cards:
                    pick = random.choice(state.middle_cards)
                    p.hand.append(pick)
                    state.middle_cards.remove(pick)
            replenish_center(state)

        state.current_round += 1

    def _play_random_round(self, state: GameState) -> None:
        """Joue un tour complet avec tous les joueurs en random."""
        for p in state.players:
            if p.hand:
                chosen = random.choice(p.hand)
                p.hand.remove(chosen)
                p.played_cards.append(chosen)
                p.current_card = chosen
        self._apply_round_rules(state)


def _player_with_sanctuary(player: Player, sanctuary: Card) -> Player:
    """Helper : retourne un joueur copié avec un sanctuaire ajouté."""
    sim = deepcopy(player)
    sim.sanctuaries.append(sanctuary)
    return sim