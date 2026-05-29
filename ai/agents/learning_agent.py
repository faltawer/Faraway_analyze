# ai/agents/learning_agent.py
from __future__ import annotations
import random
import pickle
from collections import defaultdict
from copy import deepcopy

from models.card import Card, Symbol
from models.player import Player
from models.game_state import GameState
from engine.scorer import score_player
from .lookahead_agent import LookaheadAgent


class LearningLookaheadAgent(LookaheadAgent):
    """
    Agent Q-Learning qui apprend par self-play.

    Améliorations vs version précédente :
    - Récompense propagée sur TOUTES les décisions de la partie
    - Clé d'état basée sur les symboles (pas les IDs)
    - Epsilon décroissant (moins d'exploration au fil du temps)
    - Récompense normalisée (score relatif aux adversaires)
    """

    def __init__(self, name: str, depth: int = 3, n_simulations: int = 20,
                 epsilon: float = 0.3, alpha: float = 0.1, gamma: float = 0.9):
        super().__init__(name, depth, n_simulations)
        self.Q: dict = defaultdict(lambda: defaultdict(float))
        self.alpha = alpha      # taux d'apprentissage
        self.gamma = gamma      # discount futur
        self.epsilon = epsilon  # exploration initiale
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        # Historique de la partie en cours : [(state_key, card_id), ...]
        self._history: list = []

    # ------------------------------------------------------------------ #
    #  Clé d'état — ce qui compte stratégiquement                         #
    # ------------------------------------------------------------------ #
    def _state_key(self, player: Player, state: GameState) -> tuple:
        """
        Clé compacte et pertinente :
        - Tour actuel
        - Nombre de chaque symbole disponible (dans l'ordre de pose)
        - Points déjà acquis
        - Nombre de sanctuaires
        """
        # Compter les symboles actifs
        symbol_counts = tuple(
            sum(1 for c in player.played_cards + player.sanctuaries
                for s in c.symbols if s == sym)
            for sym in Symbol
        )
        current_score, _ = score_player(player)
        return (
            state.current_round,
            symbol_counts,
            current_score,
            len(player.sanctuaries),
        )

    # ------------------------------------------------------------------ #
    #  Choix de carte — epsilon-greedy                                    #
    # ------------------------------------------------------------------ #
    def choose_card(self, player: Player, state: GameState) -> Card:
        state_key = self._state_key(player, state)

        if random.random() < self.epsilon:
            card = random.choice(player.hand)
        else:
            # Combiner Q-value + score lookahead simulé
            best_card = None
            best_value = float("-inf")
            for card in player.hand:
                q_value = self.Q[state_key][card.id]
                lookahead_score = self._evaluate_card(card, player, state)
                # Normaliser le lookahead entre 0 et 1 environ
                combined = q_value + 0.01 * lookahead_score
                if combined > best_value:
                    best_value = combined
                    best_card = card
            card = best_card or random.choice(player.hand)

        # Enregistrer la décision dans l'historique
        self._history.append((state_key, card.id))
        return card

    # ------------------------------------------------------------------ #
    #  Apprentissage en fin de partie                                     #
    # ------------------------------------------------------------------ #
    def learn_from_game(self, final_score: int, all_scores: list[int]) -> None:
        """
        Propage la récompense sur toutes les décisions de la partie.

        Récompense = score relatif aux adversaires (pas absolu)
        → encourage à battre les autres, pas juste à scorer haut
        """
        if not self._history:
            return

        avg_opponent = (sum(all_scores) - final_score) / max(len(all_scores) - 1, 1)
        # Récompense normalisée : positif si on bat la moyenne
        reward = (final_score - avg_opponent) / max(avg_opponent, 1)

        # Propagation temporelle : décision récente = récompense pleine
        # décision ancienne = récompense discountée
        discounted_reward = reward
        for state_key, card_id in reversed(self._history):
            old_q = self.Q[state_key][card_id]
            self.Q[state_key][card_id] = (
                old_q + self.alpha * (discounted_reward - old_q)
            )
            discounted_reward *= self.gamma  # discount vers le passé

        # Réinitialiser l'historique pour la prochaine partie
        self._history.clear()

        # Décroissance de epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    #  Sauvegarde / chargement                                            #
    # ------------------------------------------------------------------ #
    def save_q(self, filepath: str) -> None:
        with open(filepath, "wb") as f:
            pickle.dump({
                "Q": dict(self.Q),
                "epsilon": self.epsilon,
            }, f)
        print(f"💾 Q-table sauvegardé : {filepath}")

    def load_q(self, filepath: str) -> None:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: defaultdict(float), data["Q"])
        self.epsilon = data.get("epsilon", self.epsilon_min)
        print(f"📂 Q-table chargé : {filepath}")