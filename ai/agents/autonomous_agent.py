# ai/agents/autonomous_agent.py
from __future__ import annotations
import random
import math
import pickle
from copy import deepcopy
from collections import defaultdict

from ai.neural.encoder import round_weight
from models.card import Card
from models.player import Player
from models.game_state import GameState
from engine.scorer import score_player
from ai.base_agent import Agent


class AutonomousAgent(Agent):
    """
    IA hybride : UCB1 (apprentissage) + Probabiliste (calcul rationnel).

    Trois décisions apprises :
    1. choose_card      — quelle carte jouer depuis sa main
    2. pick_from_center — quelle carte prendre au centre (décision majeure)
    3. choose_sanctuary — quel sanctuaire garder

    UCB1 + proba sur les décisions 1 et 2.
    Proba seul sur les sanctuaires (espace trop petit pour apprendre).
    """

    def __init__(self, name: str, n_simulations: int = 200, card_file: str | None = None, load_path: str | None = None):
        super().__init__(name)
        self.n_simulations = n_simulations
        self.play_stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        self.pick_stats = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        self._history = []
        self._proba = None
        if card_file:
            self._init_proba(card_file)
        if load_path:
            self.load(load_path)

    # ------------------------------------------------------------------ #
    #  Module probabiliste                                                #
    # ------------------------------------------------------------------ #

    def _init_proba(self, card_file: str) -> None:
        from models.loader import load_cards
        from ai.agents.probabilistic_agent import ProbabilisticAgent

        proba = ProbabilisticAgent.__new__(ProbabilisticAgent)
        Agent.__init__(proba, self.name + "_proba")

        region_cards, sanctuary_cards = load_cards(card_file)
        proba.all_region_cards = {c.id: c for c in region_cards}
        proba.all_sanctuary_cards = {c.id: c for c in sanctuary_cards}
        proba.all_cards = {**proba.all_region_cards, **proba.all_sanctuary_cards}
        proba.n_samples = 200

        # Copier les méthodes utiles
        self._prob_requirements_met = proba._prob_requirements_met
        self._symbol_scarcity = proba._symbol_scarcity
        self._resource_value = proba._resource_value
        self._best_synergy_symbol = proba._best_synergy_symbol
        self._avg_sanctuary_value = proba._avg_sanctuary_value
        self._current_symbols = proba._current_symbols
        self._card_symbols = proba._card_symbols
        self._obtainable_cards = proba._obtainable_cards

        self._proba = proba
        print(f"🧠 {self.name} : module probabiliste chargé avec {len(proba.all_cards)} cartes.")

    # ------------------------------------------------------------------ #
    #  Clé d'état                                                         #
    # ------------------------------------------------------------------ #

    def _state_key(self, player: Player, state: GameState) -> tuple:
        """
        Représentation abstraite — 8 dimensions, ~48 000 états.
        Commune aux décisions play et pick.
        """
        from models.card import Symbol
        from engine.rules import card_triggers_sanctuary, count_clues

        round_phase = (
            "early" if state.current_round <= 3
            else "mid" if state.current_round <= 6
            else "late"
        )

        sym_counts = {
            sym: sum(
                c.symbols.count(sym)
                for c in player.played_cards + player.sanctuaries
            )
            for sym in [Symbol.STONE, Symbol.CHIMERA,
                        Symbol.THISTLE, Symbol.MAP]
        }
        dominant = (
            max(sym_counts, key=sym_counts.get)
            if any(sym_counts.values()) else None
        )
        dominant_val = dominant.value if dominant else "none"

        big_unmet = min(sum(
            1 for c in player.played_cards
            if len(c.activation_requirements) >= 3
            and not all(
                any(s == req
                    for played in player.played_cards + player.sanctuaries
                    for s in played.symbols)
                for req in c.activation_requirements
            )
        ), 3)

        current_score, _ = score_player(player)
        score_bucket = (current_score // 10) * 10

        high_value_in_hand = min(
            sum(1 for c in player.hand if c.points >= 10), 3
        )

        will_trigger = card_triggers_sanctuary(player)
        nb_sanctuaries = min(len(player.sanctuaries), 4)
        nb_clues = min(count_clues(player), 3)

        activatable_now = sum(
            1 for c in player.hand
            if self._can_activate(c, player)
        )
        high_potential = sum(
            c.points for c in player.hand
            if c.points >= 12 and self._can_activate_soon(c, player)
        )

        return (
            will_trigger,
            high_potential,
            activatable_now,
            big_unmet
            # dominant_val,
            # score_bucket,
            # nb_sanctuaries,
            # nb_clues,
        )

    # ------------------------------------------------------------------ #
    #  UCB1                                                               #
    # ------------------------------------------------------------------ #

    def _ucb1(
            self, stats: dict, state_key: tuple,
            card_id: int, total_visits: int
    ) -> float:
        total, visits = stats[state_key][card_id]
        if visits == 0:
            return float("inf")
        avg = total / visits
        exploration = math.sqrt(2 * math.log(total_visits + 1) / visits)
        return avg + exploration

    def _combined_score(
            self,
            card: Card,
            player: Player,
            state: GameState,
            stats: dict,
            state_key: tuple,
            candidates: list,
    ) -> float:
        """
        Score UCB1 + valeur probabiliste avec poids adaptatif.
        Plus on a d'expérience → UCB1 domine.
        """
        total_visits = sum(stats[state_key][c.id][1] for c in candidates)
        ucb1 = self._ucb1(stats, state_key, card.id, total_visits)

        if ucb1 == float("inf"):
            return ucb1

        proba_value = 0.0
        if self._proba:
            raw = self._proba._expected_value(card, player, state)
            proba_value = raw / 100.0

        total_exp = sum(stats[state_key][c.id][1] for c in candidates)
        w_proba = 0.9 if total_exp < 1000 else max(0.2, 0.8 - total_exp / 1000)
        w_ucb1 = 1.0 - w_proba

        return w_ucb1 * ucb1 + w_proba * proba_value

    def _can_activate_soon(self, card: Card, player: Player) -> bool:
        """
        Estime si la carte pourra être activée dans les prochains tours,
        en tenant compte des cartes en main et des symboles manquants.
        """
        if not card.activation_requirements:
            return True  # Pas de condition d'activation

        # Récupérer tous les symboles disponibles (cartes jouées + sanctuaires)
        available_symbols = []
        for c in player.played_cards + player.sanctuaries:
            available_symbols.extend(c.symbols)

        # Compter les symboles manquants
        missing_symbols = []
        for requirement in card.activation_requirements:
            if not any(s == requirement for s in available_symbols):
                missing_symbols.append(requirement)

        if not missing_symbols:
            return True  # Déjà activable

        # Vérifier si les symboles manquants sont dans la main du joueur
        for c in player.hand:
            for symbol in missing_symbols:
                if symbol in c.symbols:
                    missing_symbols.remove(symbol)
                    if not missing_symbols:
                        return True

        return False  # Impossible d'activer dans l'immédiat

    # ------------------------------------------------------------------ #
    #  Décision 1 — Jouer une carte depuis la main                       #
    # ------------------------------------------------------------------ #

    def _can_activate(self, card: Card, player: Player) -> bool:
        """
        Vérifie si la carte peut être activée avec les cartes jouées et les sanctuaires actuels.
        """
        if not card.activation_requirements:
            return True  # Pas de condition d'activation

        # Récupérer tous les symboles disponibles (cartes jouées + sanctuaires)
        available_symbols = []
        for c in player.played_cards + player.sanctuaries:
            available_symbols.extend(c.symbols)

        # Vérifier si tous les symboles requis sont disponibles
        for requirement in card.activation_requirements:
            if not any(s == requirement for s in available_symbols):
                return False

        return True

    def choose_card(self, player: Player, state: GameState) -> Card:
        if not player.hand:
            raise ValueError(f"{self.name} : main vide")

        state_key = self._state_key(player, state)

        def score_card(c: Card) -> float:
            total_visits = sum(
                self.pick_stats[state_key][c.id][1]
                for c in state.middle_cards
            )
            ucb1_score = self._ucb1(self.play_stats, state_key, c.id, total_visits)
            if ucb1_score == float("inf"):
                return ucb1_score
            proba_value = 0.0
            if self._proba:
                proba_value = self._proba._expected_value(c, player, state) / 100.0
            # Bonus si la carte peut être activée immédiatement
            immediate_bonus = c.points if self._can_activate(c, player) else 0
            # Pondération dynamique
            total_exp = sum(self.play_stats[state_key][cid][1] for cid in [cc.id for cc in player.hand])
            w_proba = 0.7 if total_exp < 100 else max(0.2, 0.8 - total_exp / 1000)
            w_ucb1 = 1.0 - w_proba
            return w_ucb1 * ucb1_score + w_proba * proba_value + immediate_bonus

        best = max(player.hand, key=score_card)
        self._history.append(("play", state_key, best.id))
        return best

    # ------------------------------------------------------------------ #
    #  Décision 2 — Choisir une carte au centre                          #
    # ------------------------------------------------------------------ #

    def pick_from_center(self, player: Player, state: GameState) -> Card:
        """
        Choisit quelle carte prendre au centre — décision stratégique majeure.

        Deux enjeux simultanés :
        - Valeur pour soi    : la carte complète-t-elle ma stratégie ?
        - Valeur de draft    : est-ce que je prive un adversaire d'une bonne carte ?

        UCB1 apprend quels types de cartes valent la peine d'être récupérées.
        Proba calcule la valeur espérée si on joue cette carte plus tard.
        """
        if not state.middle_cards:
            return None

        state_key = self._state_key(player, state)

        best = max(
            state.middle_cards,
            key=lambda c: self._combined_score_center(
                c, player, state, state_key
            )
        )

        self._history.append(("pick", state_key, best.id))
        return best

    def _combined_score_center(
            self,
            card: Card,
            player: Player,
            state: GameState,
            state_key: tuple,
    ) -> float:
        total_visits = sum(
            self.pick_stats[state_key][c.id][1]
            for c in state.middle_cards
        )
        ucb1 = self._ucb1(
            self.pick_stats, state_key, card.id, total_visits
        )

        if ucb1 == float("inf"):
            return ucb1

        # Initialiser sim_player ici pour qu'il soit accessible dans toute la fonction
        sim_player = deepcopy(player)
        sim_player.hand.append(card)

        proba_value = 0.0
        if self._proba:
            # Calculer la valeur probabiliste
            raw = self._proba._expected_value(card, sim_player, state)
            proba_value = raw / 100.0

        # Calculer le bonus d'activation immédiate
        immediate_activation_bonus = 0
        for c in player.hand:
            if self._can_activate_soon(c, sim_player) and not self._can_activate(c, player):
                immediate_activation_bonus += c.points

        total_exp = sum(
            self.pick_stats[state_key][c.id][1]
            for c in state.middle_cards
        )
        w_proba = 0.8 if total_exp < 200 else max(0.3, 0.9 - total_exp / 1000)
        w_ucb1 = 1.0 - w_proba

        return w_ucb1 * ucb1 + w_proba * proba_value + immediate_activation_bonus

    # def _combined_score_center(
    #     self,
    #     card: Card,
    #     player: Player,
    #     state: GameState,
    #     state_key: tuple,
    # ) -> float:
    #     """
    #     Score pour une carte du centre.
    #     On simule qu'on l'ajoute à la main avant d'évaluer sa valeur espérée.
    #     """
    #     total_visits = sum(
    #         self.pick_stats[state_key][c.id][1]
    #         for c in state.middle_cards
    #     )
    #     ucb1 = self._ucb1(
    #         self.pick_stats, state_key, card.id, total_visits
    #     )
    #
    #     if ucb1 == float("inf"):
    #         return ucb1
    #
    #     proba_value = 0.0
    #     if self._proba:
    #         # Simuler : si j'ajoute cette carte à ma main,
    #         # quelle est sa valeur espérée pour les tours futurs ?
    #         sim_player = deepcopy(player)
    #         sim_player.hand.append(card)
    #         raw = self._proba._expected_value(card, sim_player, state)
    #         proba_value = raw / 100.0
    #
    #     total_exp = sum(
    #         self.pick_stats[state_key][c.id][1]
    #         for c in state.middle_cards
    #     )
    #     w_proba = max(0.1, 0.9 - total_exp / 500)
    #     w_ucb1  = 1.0 - w_proba
    #
    #     return w_ucb1 * ucb1 + w_proba * proba_value

    # ------------------------------------------------------------------ #
    #  Décision 3 — Choisir un sanctuaire                                #
    # ------------------------------------------------------------------ #

    def _sanctuary_expected_value(
            self, sanctuary: Card, player: Player, state: GameState
    ) -> float:
        """
        Valeur espérée d'un sanctuaire :
        score immédiat + valeur ressource pour les cartes déjà posées.
        """
        score_before, _ = score_player(player)
        sim = deepcopy(player)
        sim.sanctuaries.append(sanctuary)
        score_after, _ = score_player(sim)
        return score_after - score_before

    def choose_sanctuary(self, player: Player, state: GameState) -> Card:
        if not player.sanctuaries_drawn:
            raise ValueError(f"{self.name} : pas de sanctuaire à choisir !")

        return max(
            player.sanctuaries_drawn,
            key=lambda s: self._sanctuary_expected_value(s, player, state)
        )

    def _sanctuary_value(self, sanctuary: Card, player: Player) -> int:
        sim = deepcopy(player)
        sim.sanctuaries.append(sanctuary)
        score, _ = score_player(sim)
        return score

    # ------------------------------------------------------------------ #
    #  Apprentissage                                                      #
    # ------------------------------------------------------------------ #

    def learn_from_game(
            self, final_score: int, all_scores: list
    ) -> None:
        """
        Propage la récompense avec pénalité renforcée en fin de partie.

        Signal de base :
        < 40 pts → punition
        40-60    → neutre
        > 60 pts → récompense

        Pondération temporelle :
        Tour 1 → reward × 1.0   (peut se rattraper)
        Tour 8 → reward × 4.0   (erreur impardonnable)
        """
        if not self._history:
            return

        if final_score < 40:
            base_reward = (final_score - 40) / 100
        elif final_score > 60:
            base_reward = (final_score - 60) / 100
        else:
            base_reward = 0.0

        # Reconstruire le numéro de tour depuis l'historique
        # L'historique contient 16 décisions (8 play + 8 pick)
        # On estime le tour depuis la position dans l'historique
        n = len(self._history)

        for i, (decision_type, state_key, card_id) in enumerate(reversed(self._history)):
            # Position dans la partie : 0 = fin, n-1 = début
            # Convertir en numéro de tour approximatif (1-8)
            round_approx = 8 - int(i * 8 / max(n, 1))
            round_approx = max(1, min(8, round_approx))

            # Poids selon le tour — fin de partie = plus punitive
            weight = round_weight(round_approx, mode="exponential")

            weighted_reward = base_reward * weight

            stats = (
                self.play_stats if decision_type == "play"
                else self.pick_stats
            )
            stats[state_key][card_id][0] += weighted_reward
            stats[state_key][card_id][1] += 1

        self._history.clear()

    # ------------------------------------------------------------------ #
    #  Save / Load                                                        #
    # ------------------------------------------------------------------ #

    def save(self, filepath: str) -> None:
        """
        Sauvegarde les stats play et pick.
        Le module probabiliste n'est pas sauvegardé —
        il se recharge depuis le fichier Excel.
        """
        serializable = {
            "play": {
                sk: dict(cs) for sk, cs in self.play_stats.items()
            },
            "pick": {
                sk: dict(cs) for sk, cs in self.pick_stats.items()
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(serializable, f)

        n_play = sum(
            v[1] for cs in self.play_stats.values() for v in cs.values()
        )
        n_pick = sum(
            v[1] for cs in self.pick_stats.values() for v in cs.values()
        )
        print(f"💾 {self.name} sauvegardé : {filepath} "
              f"(play={n_play}, pick={n_pick} décisions)")

    def load(self, filepath: str) -> None:
        """Charge les stats depuis un fichier — reprend là où on s'était arrêté."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        def rebuild(raw: dict) -> defaultdict:
            return defaultdict(
                lambda: defaultdict(lambda: [0.0, 0]),
                {
                    sk: defaultdict(lambda: [0.0, 0], cs)
                    for sk, cs in raw.items()
                }
            )

        self.play_stats = rebuild(data.get("play", {}))
        self.pick_stats = rebuild(data.get("pick", {}))

        n_play = sum(
            v[1] for cs in self.play_stats.values() for v in cs.values()
        )
        n_pick = sum(
            v[1] for cs in self.pick_stats.values() for v in cs.values()
        )
        print(f"📂 {self.name} chargé : {filepath} "
              f"(play={n_play}, pick={n_pick} décisions mémorisées)")

    def stats_summary(self) -> None:
        n_play = sum(
            v[1] for cs in self.play_stats.values() for v in cs.values()
        )
        n_pick = sum(
            v[1] for cs in self.pick_stats.values() for v in cs.values()
        )
        print(f"\n📊 {self.name} — statistiques :")
        print(f"   play : {len(self.play_stats)} états | {n_play} décisions")
        print(f"   pick : {len(self.pick_stats)} états | {n_pick} décisions")
        print(f"   Module proba : {'✅ actif' if self._proba else '❌ inactif'}")
