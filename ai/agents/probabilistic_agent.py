# ai/agents/probabilistic_agent.py
from __future__ import annotations
import random
from copy import deepcopy

from models.card import Card, CardType, Symbol
from models.player import Player
from models.game_state import GameState
from models.loader import load_cards
from engine.scorer import score_player
from ai.base_agent import Agent


class ProbabilisticAgent(Agent):
    """
    Agent probabiliste — connaît toutes les cartes du jeu dès le départ.

    Raisonne comme un joueur qui a étudié toutes les cartes avant la partie :
    - Sait exactement quels symboles sont encore disponibles dans le deck
    - Calcule la probabilité que ses prérequis soient remplis
    - Estime la valeur espérée de chaque carte jouable
    - Tient compte de la rareté des symboles

    Aucun apprentissage — pur calcul probabiliste à chaque décision.
    """

    def __init__(self, name: str, card_file: str, n_samples: int = 200):
        super().__init__(name)
        self.n_samples = n_samples  # précision des simulations Monte Carlo

        # Charger toutes les cartes au démarrage — connaissance complète du jeu
        region_cards, sanctuary_cards = load_cards(card_file)

        self.all_region_cards: dict[int, Card] = {
            c.id: c for c in region_cards
        }
        self.all_sanctuary_cards: dict[int, Card] = {
            c.id: c for c in sanctuary_cards
        }
        self.all_cards: dict[int, Card] = {
            **self.all_region_cards,
            **self.all_sanctuary_cards,
        }


    # ------------------------------------------------------------------ #
    #  Connaissance des cartes restantes                                  #
    # ------------------------------------------------------------------ #

    def pick_from_center(self, player, state):
        return max(state.middle_cards,
                   key=lambda c: self._expected_value(c, player, state))

    def _known_ids(self, player: Player, state: GameState) -> set[int]:
        """
        IDs de toutes les cartes déjà visibles / jouées.
        Ce qui reste = toutes les cartes - ces IDs.
        """
        known = set()

        # Sa propre main
        for c in player.hand:
            known.add(c.id)

        # Cartes jouées + sanctuaires de tout le monde
        for p in state.players:
            for c in p.played_cards + p.sanctuaries + p.sanctuaries_drawn:
                known.add(c.id)

        # Centre visible
        for c in state.middle_cards:
            known.add(c.id)

        return known

    def _unknown_region_cards(
        self, player: Player, state: GameState
    ) -> list[Card]:
        """
        Cartes région encore inconnues — potentiellement dans le deck
        ou dans les mains adverses.
        """
        known = self._known_ids(player, state)
        return [
            c for cid, c in self.all_region_cards.items()
            if cid not in known
        ]

    def _unknown_sanctuaries(
        self, player: Player, state: GameState
    ) -> list[Card]:
        """Sanctuaires encore disponibles dans le deck sanctuaire."""
        known = self._known_ids(player, state)
        return [
            c for cid, c in self.all_sanctuary_cards.items()
            if cid not in known
        ]

    def _obtainable_cards(
        self, player: Player, state: GameState
    ) -> list[Card]:
        """
        Cartes que le joueur peut encore obtenir :
        - Centre visible (certaines)
        - Cartes inconnues du deck (probabilistes)
        Exclut sa main actuelle (déjà en possession).
        """
        return state.middle_cards + self._unknown_region_cards(player, state)

    # ------------------------------------------------------------------ #
    #  Symboles disponibles                                               #
    # ------------------------------------------------------------------ #

    def _current_symbols(self, player: Player) -> list[Symbol]:
        """Symboles actuellement actifs pour ce joueur."""
        symbols = []
        for c in player.played_cards + player.sanctuaries:
            symbols.extend(c.symbols)
            if c.color:
                try:
                    symbols.append(Symbol(c.color))
                except ValueError:
                    pass
        return symbols

    def _card_symbols(self, card: Card) -> list[Symbol]:
        """Symboles apportés par une carte."""
        symbols = list(card.symbols)
        if card.color:
            try:
                symbols.append(Symbol(card.color))
            except ValueError:
                pass
        return symbols

    # ------------------------------------------------------------------ #
    #  Probabilités                                                       #
    # ------------------------------------------------------------------ #

    def _prob_requirements_met(
        self,
        requirements: list[Symbol],
        current_symbols: list[Symbol],
        future_cards: list[Card],
        turns_left: int,
    ) -> float:
        """
        Probabilité que les prérequis soient remplis d'ici la fin.

        Méthode Monte Carlo :
        - On tire `n_samples` fois `turns_left` cartes parmi les futures
        - On compte combien de tirages remplissent les prérequis restants
        """
        if not requirements:
            return 1.0

        # Vérifier les prérequis déjà remplis
        pool = current_symbols.copy()
        remaining = []
        for req in requirements:
            if req in pool:
                pool.remove(req)
            else:
                remaining.append(req)

        if not remaining:
            return 1.0  # déjà tous remplis ✅

        if not future_cards or turns_left == 0:
            return 0.0

        cards_to_draw = min(turns_left, len(future_cards))
        success = 0

        for _ in range(self.n_samples):
            sample = random.sample(future_cards, cards_to_draw)
            sample_symbols = []
            for c in sample:
                sample_symbols.extend(self._card_symbols(c))

            temp_pool = sample_symbols.copy()
            fulfilled = True
            for req in remaining:
                if req in temp_pool:
                    temp_pool.remove(req)
                else:
                    fulfilled = False
                    break

            if fulfilled:
                success += 1

        return success / self.n_samples

    def _symbol_scarcity(
        self, symbol: Symbol, player: Player, state: GameState
    ) -> float:
        """
        Proportion de cartes restantes contenant ce symbole.
        Faible = symbole rare = prérequis difficile à remplir.
        """
        remaining = self._unknown_region_cards(player, state)
        if not remaining:
            return 0.0

        count = sum(
            1 for c in remaining
            if symbol in c.symbols or c.color == symbol.value
        )
        return count / len(remaining)

    # ------------------------------------------------------------------ #
    #  Valeur espérée                                                     #
    # ------------------------------------------------------------------ #

    def _resource_value(self, card: Card, player: Player) -> float:
        """
        Points supplémentaires débloqués sur les cartes déjà posées
        grâce aux symboles apportés par cette carte.
        """
        score_before, _ = score_player(player)
        sim = deepcopy(player)
        sim.played_cards.append(card)
        score_after, _ = score_player(sim)
        return max(0.0, score_after - score_before)

    def _best_synergy_symbol(self, player: Player) -> Symbol | None:
        """
        Symbole qui débloquerait le plus de points si on l'obtenait.
        Utile pour identifier quelle ressource chercher.
        """
        best_sym = None
        best_gain = 0.0

        for sym in [Symbol.STONE, Symbol.CHIMERA,
                    Symbol.THISTLE, Symbol.MAP,
                    Symbol.RED, Symbol.BLUE,
                    Symbol.YELLOW, Symbol.GREEN]:
            # Simuler l'ajout d'une carte fictive avec ce symbole
            sim = deepcopy(player)
            fake = Card(
                id=9999, points=0, color="",
                card_type=CardType.REGION,
                symbols=[sym],
            )
            sim.played_cards.append(fake)
            score_with, _ = score_player(sim)
            score_without, _ = score_player(player)
            gain = score_with - score_without

            if gain > best_gain:
                best_gain = gain
                best_sym = sym

        return best_sym

    def _avg_sanctuary_value(
        self, player: Player, state: GameState
    ) -> float:
        """
        Valeur moyenne des sanctuaires encore disponibles.
        Utilisé pour estimer le bonus d'un sanctuaire futur.
        """
        unknown = self._unknown_sanctuaries(player, state)
        if not unknown:
            return 3.0  # valeur par défaut si inconnu
        return sum(s.points for s in unknown) / len(unknown)

    def _expected_value(
        self, card: Card, player: Player, state: GameState
    ) -> float:
        """
        Valeur espérée complète d'une carte candidate.

        Composantes :
        1. Score potentiel × P(activation)
        2. Valeur ressource (points débloqués sur cartes existantes)
        3. Bonus sanctuaire si séquence croissante
        4. Bonus synergie si apporte le symbole le plus utile
        5. Pénalité rareté si prérequis difficiles à remplir
        """
        turns_left = 8 - state.current_round
        current_sym = self._current_symbols(player)
        card_sym = self._card_symbols(card)
        future_sym = current_sym + card_sym

        future_cards = [
            c for c in self._obtainable_cards(player, state)
            if c.id != card.id
        ]

        # 1. Score potentiel × P(activation)
        prob_self = self._prob_requirements_met(
            card.activation_requirements,
            future_sym,
            future_cards,
            turns_left,
        )
        sim_player = deepcopy(player)
        sim_player.played_cards.append(card)
        _, card_scores = score_player(sim_player)
        potential_score = card_scores.get(card.id, card.points)
        value_self = potential_score * prob_self

        # 2. Valeur ressource
        value_resource = self._resource_value(card, player)

        # 3. Bonus sanctuaire
        bonus_sanctuary = 0.0
        if player.played_cards and card.id > player.played_cards[-1].id:
            avg_sanct = self._avg_sanctuary_value(player, state)
            # Pondérer par le nb de clues (map) pour estimer le gain réel
            from engine.rules import count_clues
            nb_clues = count_clues(player)
            bonus_sanctuary = avg_sanct * (1 + nb_clues * 0.3)

        # 4. Bonus synergie
        best_sym = self._best_synergy_symbol(player)
        bonus_synergy = 0.0
        if best_sym:
            card_provides = self._card_symbols(card)
            if best_sym in card_provides:
                bonus_synergy = 3.0

        # 5. Pénalité rareté des prérequis
        scarcity_penalty = 0.0
        for req in card.activation_requirements:
            scarcity = self._symbol_scarcity(req, player, state)
            if scarcity < 0.15:   # très rare
                scarcity_penalty += 3.0
            elif scarcity < 0.25: # rare
                scarcity_penalty += 1.5

        total = (value_self
                 + value_resource
                 + bonus_sanctuary
                 + bonus_synergy
                 - scarcity_penalty)

        return total

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

    # ------------------------------------------------------------------ #
    #  Interface Agent                                                    #
    # ------------------------------------------------------------------ #

    def choose_card(self, player: Player, state: GameState) -> Card:
        """
        Choisit la carte avec la meilleure valeur espérée.
        """
        if not player.hand:
            raise ValueError(f"{self.name} : main vide !")

        scored = [
            (card, self._expected_value(card, player, state))
            for card in player.hand
        ]

        # Trier par valeur espérée décroissante
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    def choose_sanctuary(self, player: Player, state: GameState) -> Card:
        """
        Choisit le sanctuaire avec la meilleure valeur espérée.
        """
        if not player.sanctuaries_drawn:
            raise ValueError(f"{self.name} : pas de sanctuaire à choisir !")

        return max(
            player.sanctuaries_drawn,
            key=lambda s: self._sanctuary_expected_value(s, player, state)
        )