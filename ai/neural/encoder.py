# ai/neural/encoder.py
"""
Encodage de l'état du jeu en vecteur numérique pour le réseau de neurones.

Principe : tout ce que le joueur sait à l'instant T est encodé
en un vecteur fixe de taille INPUT_DIM.

Structure du vecteur :
  [0]       : tour actuel normalisé (0-1)
  [1]       : score actuel normalisé
  [2]       : nb sanctuaires normalisé
  [3]       : nb cartes en main normalisé
  [4..13]   : symboles actifs (counts, plafonnés à 5, normalisés)
  [14..23]  : symboles de la carte candidate
  [24..33]  : prérequis de la carte candidate (one-hot counts)
  [34..43]  : effet conditionnel de la carte candidate
  [44]      : points de la carte candidate normalisés
  [45]      : probabilité estimée d'activation (0-1)
  [46]      : valeur ressource nsormalisée
  [47]      : bonus sanctuaire (0 ou 1)
  [48]      : rareté moyenne des prérequis (0-1)
  [49]      : phase de jeu (0=early, 0.5=mid, 1=late)

Total : INPUT_DIM = 54
"""
from __future__ import annotations
import numpy as np
from models.card import Card, Symbol, multiplicator_effect
from models.player import Player
from models.game_state import GameState

# Ordre fixe des symboles — doit être cohérent partout
SYMBOLS = [
    Symbol.STONE, Symbol.CHIMERA, Symbol.THISTLE, Symbol.MAP,
    Symbol.RED, Symbol.BLUE, Symbol.YELLOW, Symbol.GREEN,
    Symbol.NIGHT, Symbol.DAY,
]

INPUT_DIM = 54

def _symbol_counts(cards: list[Card]) -> np.ndarray:
    """Compte les occurrences de chaque symbole dans une liste de cartes."""
    counts = np.zeros(len(SYMBOLS), dtype=np.float32)
    for card in cards:
        for sym in card.symbols:
            if sym in SYMBOLS:
                counts[SYMBOLS.index(sym)] += 1
        if card.color:
            try:
                s = Symbol(card.color)
                if s in SYMBOLS:
                    counts[SYMBOLS.index(s)] += 1
            except ValueError:
                pass
    return np.clip(counts / 5.0, 0, 1)  # normaliser


def _req_counts(requirements: list[Symbol]) -> np.ndarray:
    """Encode les prérequis d'une carte."""
    counts = np.zeros(len(SYMBOLS), dtype=np.float32)
    for req in requirements:
        if req in SYMBOLS:
            counts[SYMBOLS.index(req)] += 1
    return np.clip(counts / 4.0, 0, 1)


def _effect_counts(effects: list[multiplicator_effect]) -> np.ndarray:
    """Encode les effets conditionnels d'une carte."""
    counts = np.zeros(len(SYMBOLS), dtype=np.float32)
    for eff in effects:
        if eff == multiplicator_effect.FOUR_COLORS:
            # Marquer les 4 couleurs
            for s in [Symbol.RED, Symbol.BLUE, Symbol.YELLOW, Symbol.GREEN]:
                counts[SYMBOLS.index(s)] += 0.5
        else:
            try:
                s = Symbol(eff.value)
                if s in SYMBOLS:
                    counts[SYMBOLS.index(s)] += 1
            except ValueError:
                pass
    return np.clip(counts / 4.0, 0, 1)

def round_weight(round_num: int, mode: str = "exponential") -> float:
    """
    Calcule le poids d'une décision selon son tour.

    Trois modes disponibles :

    'linear'      : poids augmente linéairement (1 → 2)
    'exponential' : poids augmente exponentiellement (1 → 4)
    'step'        : faible jusqu'au tour 5, fort ensuite

    Tour    linear  exponential  step
    1       1.00    1.00         0.50
    2       1.14    1.23         0.50
    3       1.29    1.52         0.50
    4       1.43    1.87         0.50
    5       1.57    2.30         1.00
    6       1.71    2.83         2.00
    7       1.86    3.48         3.00
    8       2.00    4.00         4.00
    """
    t = (round_num - 1) / 7.0  # normaliser 0→1

    if mode == "linear":
        return 1.0 + t

    elif mode == "exponential":
        # Croissance exponentielle : tour 1 = 1×, tour 8 = 4×
        return float(4 ** t)

    elif mode == "step":
        # Paliers : early/mid/late
        if round_num <= 4:
            return 0.5
        elif round_num <= 6:
            return 2.0
        else:
            return 4.0

    return 1.0

def _prob_activation(
    card: Card,
    active_symbols: list[Symbol],
    future_cards: list[Card],
    turns_left: int,
    n_samples: int = 50,
) -> float:
    """Estimation rapide de la probabilité d'activation."""
    import random
    if not card.activation_requirements:
        return 1.0

    pool = active_symbols.copy()
    remaining = []
    for req in card.activation_requirements:
        if req in pool:
            pool.remove(req)
        else:
            remaining.append(req)

    if not remaining:
        return 1.0
    if not future_cards or turns_left == 0:
        return 0.0

    cards_to_draw = min(turns_left, len(future_cards))
    success = 0
    for _ in range(n_samples):
        sample = random.sample(future_cards, min(cards_to_draw, len(future_cards)))
        sample_syms = []
        for c in sample:
            sample_syms.extend(c.symbols)
            if c.color:
                try:
                    sample_syms.append(Symbol(c.color))
                except ValueError:
                    pass
        temp = sample_syms.copy()
        ok = True
        for req in remaining:
            if req in temp:
                temp.remove(req)
            else:
                ok = False
                break
        if ok:
            success += 1
    return success / n_samples


def state_to_vector(
    player: Player,
    state: GameState,
    candidate: Card,
    future_cards: list[Card] | None = None,
) -> np.ndarray:
    """
    Encode l'état complet + carte candidate en vecteur INPUT_DIM = 50.

    Parameters
    ----------
    player    : joueur courant
    state     : état du jeu
    candidate : carte qu'on envisage de jouer
    future_cards : cartes encore disponibles (deck + centre), None = inféré
    """
    from engine.scorer import score_player, collect_symbols
    from engine.rules import card_triggers_sanctuary, count_clues
    from copy import deepcopy

    vec = np.zeros(INPUT_DIM, dtype=np.float32)

    # [0] Tour normalisé
    vec[0] = (state.current_round - 1) / 7.0

    # [1] Score actuel normalisé
    current_score, _ = score_player(player)
    vec[1] = min(current_score / 100.0, 1.0)

    # [2] Nb sanctuaires normalisé
    vec[2] = min(len(player.sanctuaries) / 4.0, 1.0)

    # [3] Nb cartes en main normalisé
    vec[3] = min(len(player.hand) / 5.0, 1.0)

    # [4..13] Symboles actifs (cartes jouées + sanctuaires)
    active_syms = collect_symbols(player.played_cards + player.sanctuaries)
    active_counts = np.zeros(len(SYMBOLS), dtype=np.float32)
    for s in active_syms:
        if s in SYMBOLS:
            active_counts[SYMBOLS.index(s)] += 1
    vec[4:14] = np.clip(active_counts / 5.0, 0, 1)

    # [14..23] Symboles de la carte candidate
    vec[14:24] = _symbol_counts([candidate])

    # [24..33] Prérequis de la carte candidate
    vec[24:34] = _req_counts(candidate.activation_requirements)

    # [34..43] Effet conditionnel de la carte candidate
    vec[34:44] = _effect_counts(candidate.multiplicator_effect)

    # [44] Points de la carte candidate normalisés
    vec[44] = min(candidate.points / 25.0, 1.0)

    # [45] Probabilité d'activation estimée
    turns_left = 8 - state.current_round
    if future_cards is None:
        future_cards = state.deck + state.middle_cards
    vec[45] = _prob_activation(
        candidate, active_syms, future_cards, turns_left
    )

    # [46] Valeur ressource normalisée
    # (combien de points supplémentaires cette carte débloque sur les cartes posées)
    sim = deepcopy(player)
    sim.played_cards.append(candidate)
    new_score, _ = score_player(sim)
    resource_value = max(0, new_score - current_score)
    vec[46] = min(resource_value / 30.0, 1.0)

    # [47] Bonus sanctuaire (séquence croissante ?)
    vec[47] = float(
        len(player.played_cards) > 0 and
        candidate.id > player.played_cards[-1].id
    )

    # [48] Rareté moyenne des prérequis
    if candidate.activation_requirements and future_cards:
        rarities = []
        for req in candidate.activation_requirements:
            count = sum(
                1 for c in future_cards
                if req in c.symbols or c.color == req.value
            )
            rarities.append(count / max(len(future_cards), 1))
        vec[48] = float(np.mean(rarities))
    else:
        vec[48] = 1.0  # pas de prérequis = pas de rareté

    # [49] Phase de jeu
    if state.current_round <= 3:
        vec[49] = 0.0
    elif state.current_round <= 6:
        vec[49] = 0.5
    else:
        vec[49] = 1.0

    # [50] Ratio cartes activables / cartes en main
    activatable = sum(
        1 for c in player.hand
        if not c.activation_requirements
        or all(req in [s for pc in player.played_cards + player.sanctuaries
                       for s in pc.symbols]
               for req in c.activation_requirements)
    )
    vec[50] = activatable / max(len(player.hand), 1)
    
    # [51] Score max possible si on joue la meilleure carte restante
    if player.hand:
        best_potential = max(c.points for c in player.hand)
        vec[51] = min(best_potential / 25.0, 1.0)
    else:
        vec[51] = 0.0

    future_synergy = sum(
        1 for c in future_cards
        if any(sym in c.symbols for sym in candidate.symbols)
    )

    vec[52] = min(future_synergy / 20.0, 1.0)
    shared_synergy = sum(
        1 for sym in candidate.symbols
        if sym in active_syms
    )
    vec[53] = shared_synergy / 5.0
    return vec


def batch_encode(
    player: Player,
    state: GameState,
    candidates: list[Card],
    future_cards: list[Card] | None = None,
) -> np.ndarray:
    """
    Encode un batch de cartes candidates en matrice (N, INPUT_DIM).
    Plus efficace que d'appeler state_to_vector N fois.
    """
    if future_cards is None:
        future_cards = state.deck + state.middle_cards
    return np.stack([
        state_to_vector(player, state, c, future_cards)
        for c in candidates
    ])
