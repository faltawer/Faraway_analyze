# ai/neural/data_collector.py
"""
Génère des données d'entraînement pour le réseau de neurones.

Principe :
  On simule N parties avec des agents existants (Greedy, Proba, Random).
  À chaque décision (choose_card, pick_from_center), on enregistre :
    - Le vecteur d'état au moment de la décision
    - Le score final obtenu à la fin de la partie
  
  Le réseau apprend à prédire ce score final depuis le vecteur d'état.

Format des données :
  X : np.ndarray (N_decisions, INPUT_DIM)
  y : np.ndarray (N_decisions,)  — score final de la partie
"""
from __future__ import annotations
import random
import numpy as np
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
from ai.neural.encoder import state_to_vector, batch_encode, round_weight
from ai.base_agent import Agent


class DataCollector:
    """
    Collecte des données d'entraînement en simulant des parties complètes.

    À chaque décision d'un agent "cible", on capture :
      (vecteur_état_avant_décision, score_final_après_8_tours)

    On peut utiliser n'importe quel agent comme "joueur cible" :
    - Greedy/Proba donnent des données de bonne qualité
    - Random donne de la diversité
    - Mieux : mix des deux
    """

    def __init__(self, card_file: str):
        from models.loader import load_cards
        region_cards, sanctuary_cards = load_cards(card_file)
        self.all_region_cards    = region_cards
        self.all_sanctuary_cards = sanctuary_cards
        self.card_file = card_file

        self.X: list[np.ndarray] = []
        self.y: list[float]      = []
        self.weights: list[float] = []

    def collect(
        self,
        n_games: int,
        agents_factory,
        target_agent_name: str,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Lance n_games parties et collecte les décisions de l'agent cible.

        Parameters
        ----------
        n_games            : nombre de parties à simuler
        agents_factory     : fonction () → list[Agent]
        target_agent_name  : nom de l'agent dont on collecte les décisions
        verbose            : afficher la progression

        Returns
        -------
        X : np.ndarray (N, INPUT_DIM)
        y : np.ndarray (N,)
        """
        for i in range(n_games):
            self._run_game(agents_factory, target_agent_name)

            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1:5d}/{n_games} parties | "
                      f"{len(self.X):6d} exemples collectés")

        X = np.stack(self.X)
        y = np.array(self.y, dtype=np.float32)
        W = np.array(self.weights, dtype=np.float32)

        if verbose:
            print(f"\n✅ Collecte terminée : {len(X)} exemples")
            print(f"   Score moyen : {y.mean():.1f} ± {y.std():.1f}")
            print(f"   Score min/max : {y.min():.0f} / {y.max():.0f}")

        return X, y, W

    def _setup(self, agents):
        random.shuffle(self.all_region_cards)
        random.shuffle(self.all_sanctuary_cards)

        region_cards    = self.all_region_cards.copy()
        sanctuary_cards = self.all_sanctuary_cards.copy()

        players = [Player(name=a.name) for a in agents]
        state   = GameState(
            players       = players,
            deck          = region_cards,
            middle_cards  = [],
            sanctuary_deck= sanctuary_cards,
            current_round = 1,
        )
        state.middle_cards = [state.deck.pop(0) for _ in range(len(players) + 1)]
        for _ in range(3):
            for p in players:
                if state.deck:
                    p.hand.append(state.deck.pop(0))
        return players, state

    def _run_game(self, agents_factory, target_name: str) -> None:
        """Joue une partie complète et collecte les décisions de target_name."""
        agents  = agents_factory()
        players, state = self._setup(agents)

        # Buffer : décisions en attente du score final
        pending: list[tuple[np.ndarray, int]] = []

        for _ in range(8):
            if state.current_round > 1:
                replenish_center(state)

            agent_by_name = {a.name: a for a in agents}

            # --- choose_card ---
            for agent, player in zip(agents, players):
                future_cards = state.deck + state.middle_cards

                if agent.name == target_name:
                    card = agent.choose_card(player, state)
                    vec = state_to_vector(player, state, card, future_cards)
                    pending.append((vec, state.current_round))  # ← ajouter round
                else:
                    card = agent.choose_card(player, state)

                player.hand.remove(card)
                player.played_cards.append(card)
                player.current_card = card

            order = get_play_order(players)

            # --- pick_from_center ---
            if state.current_round < 8:
                for player in order:
                    if state.middle_cards:
                        agent        = agent_by_name[player.name]
                        future_cards = state.deck + state.middle_cards

                        if agent.name == target_name:
                            pick = agent.pick_from_center(player, state)
                            vec  = state_to_vector(player, state, pick, future_cards)
                            pending.append((vec, state.current_round))
                        else:
                            pick = agent.pick_from_center(player, state)

                        if pick:
                            player.hand.append(pick)
                            state.middle_cards.remove(pick)

            # --- Sanctuaires ---
            if state.current_round > 1:
                for player in order:
                    draw_sanctuaries(player, state)
                for player in order:
                    if player.sanctuaries_drawn:
                        agent  = agent_by_name[player.name]
                        chosen = agent.choose_sanctuary(player, state)
                        if chosen:
                            player.sanctuaries.append(chosen)
                            player.sanctuaries_drawn.remove(chosen)
                    discard_excess_sanctuaries(player, state)

            state.current_round += 1

        # Score final du joueur cible
        target_player = next(p for p in players if p.name == target_name)
        final_score, _ = score_player(target_player)

        # Enregistrer tous les exemples de cette partie
        for vec, round_num in pending:
            self.X.append(vec)
            self.y.append(float(final_score))
            weight = round_weight(round_num, mode="exponential")
            self.weights.append(weight)

    def save(self, path: str) -> None:
        """Sauvegarde les données collectées."""
        X = np.stack(self.X)
        y = np.array(self.y, dtype=np.float32)
        np.savez_compressed(path, X=X, y=y)
        print(f"💾 Données sauvegardées : {path} ({len(X)} exemples)")

    @staticmethod
    def load(path: str) -> tuple[np.ndarray, np.ndarray]:
        """Charge des données sauvegardées."""
        data = np.load(path)
        print(f"📂 Données chargées : {path} ({len(data['X'])} exemples)")
        return data["X"], data["y"]
