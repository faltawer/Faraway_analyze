# ai/agents/neural_agent.py
"""
Agent qui utilise un réseau de neurones pour choisir ses cartes.

Stratégie :
  Pour chaque carte jouable, on calcule le vecteur d'état (encoder.py),
  on passe ce vecteur dans le réseau (model.py), et on choisit
  la carte dont le score prédit est le plus élevé.

Optionnel : fallback probabiliste si le réseau n'est pas chargé.
"""
from __future__ import annotations
import numpy as np
from copy import deepcopy

from models.card import Card
from models.player import Player
from models.game_state import GameState
from engine.scorer import score_player
from ai.base_agent import Agent
from ai.neural.encoder import batch_encode


class NeuralAgent(Agent):
    """
    Agent neuronal pour Faraway.

    Utilise FarawayNet pour prédire le score final associé
    à chaque carte jouable, et choisit celle dont le score est le plus élevé.

    Paramètres
    ----------
    name       : nom de l'agent
    model_path : chemin vers le fichier .pt du modèle entraîné
    card_file  : chemin vers l'Excel (pour le fallback probabiliste)
    epsilon    : exploration aléatoire résiduelle (0 = pure exploitation)
    """

    def __init__(
            self,
            name: str,
            model_path: str | None = None,
            card_file: str | None = None,
            epsilon: float = 0.05,
            risk_bonus: float = 0.3,
    ):
        super().__init__(name)
        self.epsilon = epsilon
        self.risk_bonus = risk_bonus
        self._model  = None
        self._proba  = None



        # Charger le réseau
        if model_path:
            self._load_model(model_path)

        # Fallback probabiliste pour choose_sanctuary + cas sans modèle
        if card_file:
            self._init_proba(card_file)

    # ------------------------------------------------------------------ #
    #  Chargement                                                         #
    # ------------------------------------------------------------------ #

    def _load_model(self, path: str) -> None:
        from ai.neural.model import FarawayNet
        self._model = FarawayNet.load(path)
        # print(f"🧠 {self.name} : réseau neuronal chargé ({path})")

    def _init_proba(self, card_file: str) -> None:
        """Initialise le module probabiliste silencieusement."""
        from models.loader import load_cards
        from ai.agents.probabilistic_agent import ProbabilisticAgent

        proba = ProbabilisticAgent.__new__(ProbabilisticAgent)
        Agent.__init__(proba, self.name + "_proba")

        region_cards, sanctuary_cards = load_cards(card_file)
        proba.all_region_cards    = {c.id: c for c in region_cards}
        proba.all_sanctuary_cards = {c.id: c for c in sanctuary_cards}
        proba.all_cards = {**proba.all_region_cards, **proba.all_sanctuary_cards}
        proba.n_samples = 80

        self._proba = proba

    # ------------------------------------------------------------------ #
    #  Décision 1 — Jouer une carte                                      #
    # ------------------------------------------------------------------ #

    def choose_card(self, player: Player, state: GameState) -> Card:
        """
        Choisit la carte depuis la main dont le score prédit est le plus élevé.
        """
        if not player.hand:
            raise ValueError(f"{self.name} : main vide")

        import random
        if random.random() < self.epsilon:
            return random.choice(player.hand)

        return self._best_card(player.hand, player, state)

    # ------------------------------------------------------------------ #
    #  Décision 2 — Piocher au centre                                    #
    # ------------------------------------------------------------------ #

    def pick_from_center(self, player: Player, state: GameState) -> Card:
        """
        Choisit la carte du centre dont le score prédit est le plus élevé.
        """
        if not state.middle_cards:
            return None

        import random
        if random.random() < self.epsilon:
            return random.choice(state.middle_cards)

        return self._best_card(state.middle_cards, player, state)

    # ------------------------------------------------------------------ #
    #  Décision 3 — Sanctuaire                                           #
    # ------------------------------------------------------------------ #

    def choose_sanctuary(self, player: Player, state: GameState) -> Card:
        """
        Sanctuaire : utilise le module probabiliste (plus adapté que le réseau).
        """
        if not player.sanctuaries_drawn:
            return None

        if self._proba:
            return self._proba.choose_sanctuary(player, state)

        # Fallback simple : score immédiat
        return max(
            player.sanctuaries_drawn,
            key=lambda s: self._sanctuary_value(s, player)
        )

    def _sanctuary_value(self, sanctuary: Card, player: Player) -> int:
        sim = deepcopy(player)
        sim.sanctuaries.append(sanctuary)
        score, _ = score_player(sim)
        return score

    # ------------------------------------------------------------------ #
    #  Cœur : prédiction par le réseau                                   #
    # ------------------------------------------------------------------ #

    def _best_card(self, candidates, player, state):
        if self._model is None:
            # fallback...
            pass

        future_cards = state.deck + state.middle_cards
        X = batch_encode(player, state, candidates, future_cards)
        scores = self._model.predict_batch(X)

        # Bonus de risque : favoriser les cartes à fort potentiel
        # même si leur score prédit est incertain
        if self.risk_bonus > 0:
            for i, card in enumerate(candidates):
                # Bonus proportionnel aux points de la carte
                # → encourage à jouer les grosses cartes tôt
                phase = (state.current_round - 1) / 7.0
                # Le bonus diminue en fin de partie
                # (on ne prend plus de risques au tour 7-8)
                time_decay = max(0, 1.0 - phase * 1.5)
                scores[i] += self.risk_bonus * card.points * time_decay / 25.0

        best_idx = int(np.argmax(scores))
        return candidates[best_idx]

    def score_candidates(
        self, candidates: list[Card], player: Player, state: GameState
    ) -> list[tuple[Card, float]]:
        """
        Retourne la liste (carte, score_prédit) triée par score décroissant.
        Utile pour débugger et comprendre les décisions du réseau.
        """
        if self._model is None:
            return [(c, float(c.points)) for c in candidates]

        future_cards = state.deck + state.middle_cards
        X = batch_encode(player, state, candidates, future_cards)
        scores = self._model.predict_batch(X)

        return sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True
        )
