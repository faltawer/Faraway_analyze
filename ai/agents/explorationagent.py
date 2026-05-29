import random
import pickle
import os
from collections import defaultdict
from copy import deepcopy
from models.card import Card, CardType
from models.player import Player
from models.game_state import GameState
from ai.base_agent import Agent
from engine.scorer import score_player

class ExplorationAgent(Agent):
    def __init__(self, name: str, exploration_rate: float = 0.8, save_path: str = "winning_strategies.pkl"):
        super().__init__(name)
        self.exploration_rate = exploration_rate
        self.save_path = save_path
        self.winning_strategies = []
        self.current_strategy = None
        self.load_strategies()

    def load_strategies(self):
        try:
            with open(self.save_path, "rb") as f:
                all_strategies = pickle.load(f)

            # Filtrer et corriger les stratégies
            self.winning_strategies = []
            for strategy in all_strategies:
                if strategy.get("score", 0) > 90 and "initial_state" in strategy:
                    self.winning_strategies.append(strategy)

            print(f"📂 {len(self.winning_strategies)} stratégies haut-niveau chargées.")
        except FileNotFoundError:
            self.winning_strategies = []

    def save_strategies(self):
        with open(self.save_path, "wb") as f:
            pickle.dump(self.winning_strategies, f)
        print(f"💾 {len(self.winning_strategies)} stratégies sauvegardées.")

    def start_new_game(self, player: Player, state: GameState):
        if not self.winning_strategies:
            self.current_strategy = None
            return

        current_initial_state = {
            "hand": sorted([card.id for card in player.hand]),
            "played_cards": sorted([card.id for card in player.played_cards]),
            "sanctuaries": sorted([s.id for s in player.sanctuaries]),
            "round": state.current_round,
        }

        def similarity(state1, state2):
            hand_sim = len(set(state1["hand"]) & set(state2["hand"])) / max(len(state1["hand"]), 1)
            played_sim = len(set(state1["played_cards"]) & set(state2["played_cards"])) / max(len(state1["played_cards"]), 1)
            sanct_sim = len(set(state1["sanctuaries"]) & set(state2["sanctuaries"])) / max(len(state1["sanctuaries"]), 1)
            round_sim = 1.0 if state1["round"] == state2["round"] else 0.0
            return 0.4 * hand_sim + 0.3 * played_sim + 0.2 * sanct_sim + 0.1 * round_sim

        best_strategy = None
        best_similarity = 0.0
        for strategy in self.winning_strategies:
            sim = similarity(current_initial_state, strategy["initial_state"])
            if sim > best_similarity:
                best_similarity = sim
                best_strategy = strategy

        if best_similarity > 0.6:
            self.current_strategy = deepcopy(best_strategy)
            print(f"🎯 Stratégie choisie (similarité: {best_similarity:.2f}, score: {best_strategy['score']})")
        else:
            self.current_strategy = None

        # Décroissance de l'exploration
        self.exploration_rate = max(0.1, self.exploration_rate * 0.99)

    def choose_card(self, player: Player, state: GameState) -> Card:
        if not player.hand:
            raise ValueError(f"{self.name}: main vide")

        if (self.current_strategy and
            "decisions" in self.current_strategy and
            "played_cards" in self.current_strategy["decisions"] and
            self.current_strategy["decisions"]["played_cards"]):
            next_card_id = self.current_strategy["decisions"]["played_cards"][0]
            for card in player.hand:
                if card.id == next_card_id:
                    self.current_strategy["decisions"]["played_cards"].pop(0)
                    return card

        # Heuristique d'exploration dirigée
        scored_cards = []
        for card in player.hand:
            score = card.points
            if card.card_type == CardType.SANCTUARY and player.sanctuaries and card.id > player.sanctuaries[-1].id:
                score *= 2.0
            for req in card.activation_requirements:
                if not any(req in c.symbols for c in player.played_cards + player.sanctuaries + player.hand):
                    score *= 0.7
            scored_cards.append((card, score))

        scored_cards.sort(key=lambda x: x[1], reverse=True)
        return scored_cards[0][0] if scored_cards else random.choice(player.hand)

    def pick_from_center(self, player: Player, state: GameState) -> Card | None:
        if not state.middle_cards:
            return None

        if (self.current_strategy and
            "decisions" in self.current_strategy and
            "picked_cards" in self.current_strategy["decisions"] and
            self.current_strategy["decisions"]["picked_cards"]):
            next_card_id = self.current_strategy["decisions"]["picked_cards"][0]
            for card in state.middle_cards:
                if card.id == next_card_id:
                    self.current_strategy["decisions"]["picked_cards"].pop(0)
                    return card

        chosen_card = random.choice(state.middle_cards)
        if not self.current_strategy:
            self.current_strategy = {
                "initial_state": {
                    "hand": sorted([card.id for card in player.hand]),
                    "played_cards": sorted([card.id for card in player.played_cards]),
                    "sanctuaries": sorted([s.id for s in player.sanctuaries]),
                    "round": state.current_round,
                },
                "decisions": {
                    "played_cards": [card.id for card in player.played_cards],
                    "picked_cards": [chosen_card.id],
                    "sanctuaries": [s.id for s in player.sanctuaries],
                },
            }
        elif "decisions" in self.current_strategy and "picked_cards" in self.current_strategy["decisions"]:
            self.current_strategy["decisions"]["picked_cards"].append(chosen_card.id)

        return chosen_card

    def choose_sanctuary(self, player: Player, state: GameState) -> Card | None:
        if not player.sanctuaries_drawn:
            return None

        if (self.current_strategy and
                "decisions" in self.current_strategy and
                "sanctuaries" in self.current_strategy["decisions"] and
                len(self.current_strategy["decisions"]["sanctuaries"]) > len(player.sanctuaries)):

            next_sanctuary_id = self.current_strategy["decisions"]["sanctuaries"][len(player.sanctuaries)]

            for sanctuary in player.sanctuaries_drawn:
                if sanctuary.id == next_sanctuary_id:
                    return sanctuary

        chosen_sanctuary = random.choice(player.sanctuaries_drawn)
        if self.current_strategy and "decisions" in self.current_strategy and "sanctuaries" in self.current_strategy["decisions"]:
            self.current_strategy["decisions"]["sanctuaries"].append(chosen_sanctuary.id)
        return chosen_sanctuary

    def learn_from_game(self, final_score: int, player: Player, state: GameState):
        if final_score > 80 and self.current_strategy:
            if "initial_state" not in self.current_strategy:
                self.current_strategy["initial_state"] = {
                    "hand": sorted([card.id for card in player.hand]),
                    "played_cards": sorted([card.id for card in player.played_cards]),
                    "sanctuaries": sorted([s.id for s in player.sanctuaries]),
                    "round": state.current_round,
                }
            self.current_strategy["score"] = final_score
            self.winning_strategies.append(self.current_strategy)
            self.save_strategies()
            print(f"🎉 Stratégie gagnante enregistrée (score: {final_score})")