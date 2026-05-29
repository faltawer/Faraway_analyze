# ai/agents/human_agent.py
from __future__ import annotations
import threading
from models.card import Card
from models.player import Player
from models.game_state import GameState
from ai.base_agent import Agent


class HumanAgent(Agent):
    def __init__(self, name: str):
        super().__init__(name)
        self._card_event      = threading.Event()
        self._center_event    = threading.Event()
        self._sanctuary_event = threading.Event()
        self._chosen_card:      Card | None = None
        self._chosen_center:    Card | None = None
        self._chosen_sanctuary: Card | None = None

    def choose_card(self, player, state):
        self._card_event.clear()
        self._chosen_card = None
        self._card_event.wait()
        return self._chosen_card

    def set_card_choice(self, card):
        self._chosen_card = card
        self._card_event.set()

    def pick_from_center(self, player, state):
        self._center_event.clear()
        self._chosen_center = None
        self._center_event.wait()
        return self._chosen_center

    def set_center_choice(self, card):
        self._chosen_center = card
        self._center_event.set()

    def choose_sanctuary(self, player, state):
        self._sanctuary_event.clear()
        self._chosen_sanctuary = None
        self._sanctuary_event.wait()
        return self._chosen_sanctuary

    def set_sanctuary_choice(self, card):
        self._chosen_sanctuary = card
        self._sanctuary_event.set()