# ai/agents/aggressive_agent.py
from __future__ import annotations
from models.card import Card
from models.player import Player
from models.game_state import GameState
from ai.agents.probabilistic_agent import ProbabilisticAgent
from ai.base_agent import Agent


class AggressiveAgent(ProbabilisticAgent):
    """
    Variante agressive du ProbabilisticAgent.

    Stratégie : favorise les cartes à fort potentiel même avec
    des prérequis difficiles — prend des risques calculés.

    Utile pour générer des données d'entraînement montrant
    au réseau que les hauts scores nécessitent de la prise de risque.
    """

    def __init__(self, name: str, card_file: str):
        super().__init__(name, card_file)
        self.risk_factor = 1.8  # amplifie les cartes à fort potentiel

    def _expected_value(
        self, card: Card, player: Player, state: GameState
    ) -> float:
        """
        Valeur espérée amplifiée pour les cartes à fort potentiel.

        Formule : valeur_base ^ risk_factor
        → Une carte à 10 pts espérés devient 10^1.8 = 63
        → Une carte à 3 pts espérés devient 3^1.8 = 7
        → L'écart est amplifié : l'agent préfère fortement les grosses cartes
        """
        base = super()._expected_value(card, player, state)
        # Amplifier les valeurs positives, atténuer les négatives
        if base > 0:
            return base ** self.risk_factor
        return base