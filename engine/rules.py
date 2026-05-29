from __future__ import annotations
from models.card import Card
from models.player import Player
from models.game_state import GameState
import random


def get_play_order(players: list[Player]) -> list[Player]:
    """
    Ordre de jeu : du plus petit au plus grand ID de carte jouée ce tour.
    """
    return sorted(players, key=lambda p: p.current_card.id)


def card_triggers_sanctuary(player: Player) -> bool:
    """
    Une carte déclenche un sanctuaire si sa valeur est supérieure
    à la carte précédente (ordre croissant).
    """
    if len(player.played_cards) < 2:
        return False
    prev = player.played_cards[-2]
    curr = player.played_cards[-1]
    return curr.id > prev.id


def count_clues(player: Player) -> int:
    """
    Compte les symboles 'map' du joueur (indices).
    Chaque indice permet de piocher un sanctuaire supplémentaire.
    """
    from models.card import Symbol
    return sum(
        c.symbols.count(Symbol.MAP)
        for c in player.played_cards + player.sanctuaries
    )


def draw_sanctuaries(player: Player, state: GameState) -> None:
    """
    Tire les sanctuaires auxquels le joueur a droit ce tour.
    1 sanctuaire de base + 1 par indice (map) possédé.
    """
    if not card_triggers_sanctuary(player):
        return

    n = 1 + count_clues(player)
    n = min(n, len(state.sanctuary_deck))

    player.sanctuaries_drawn = [
        state.sanctuary_deck.pop(0) for _ in range(n)
    ]


def choose_sanctuary(player: Player, state: GameState) -> None:
    """
    Le joueur choisit un sanctuaire parmi ceux tirés.
    Pour l'instant : choix aléatoire (sera remplacé par l'Agent).
    """
    if not player.sanctuaries_drawn:
        return
    chosen = random.choice(player.sanctuaries_drawn)
    player.sanctuaries.append(chosen)
    player.sanctuaries_drawn.clear()


def discard_excess_sanctuaries(player: Player, state: GameState) -> None:
    """
    Défausse les sanctuaires non choisis et l'excédent éventuel.
    """
    # Défausser ce qui n'a pas été choisi ce tour
    while player.sanctuaries_drawn:
        state.sanctuary_deck.append(player.sanctuaries_drawn.pop())

    # Défausser l'excédent selon la vraie règle
    max_allowed = max_sanctuaries_allowed(player)
    while len(player.sanctuaries) > max_allowed:
        state.sanctuary_deck.append(player.sanctuaries.pop())


def max_sanctuaries_allowed(player: Player) -> int:
    """
    Nombre maximum de sanctuaires = nombre de fois où
    la séquence de cartes était croissante.

    Exemple : [3, 7, 5, 12, 8] → croissant aux positions (3→7) et (5→12)
              → max 2 sanctuaires
    """
    if len(player.played_cards) < 2:
        return 0
    return sum(
        1 for i in range(1, len(player.played_cards))
        if player.played_cards[i].id > player.played_cards[i - 1].id
    )


def replenish_center(state: GameState) -> None:
    expected_cards = len(state.players) + 1  # N+1
    missing_cards = expected_cards - len(state.middle_cards)

    if missing_cards > 0 and state.deck:
        n = min(missing_cards, len(state.deck))
        state.middle_cards.extend([state.deck.pop(0) for _ in range(n)])
    # Sinon, le centre a moins de cartes (fin de partie proche)