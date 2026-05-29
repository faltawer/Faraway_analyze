from __future__ import annotations
from models.card import Card, Symbol, multiplicator_effect
from models.player import Player
def collect_symbols(cards: list) -> list:
    symbols = []
    for card in cards:
        symbols.extend(card.symbols)
        if card.color:
            try:
                symbols.append(Symbol(card.color))
            except ValueError:
                pass
    return symbols


def count_full_sets(requirements: list, pool: list) -> int:
    if not requirements:
        return 0
    return min(pool.count(r) // requirements.count(r) for r in set(requirements))

def conditional_to_symbol(effect: multiplicator_effect) -> Symbol | None:
    """
    Convertit un ConditionalEffect en Symbol équivalent pour le comptage.
    Retourne None pour les effets spéciaux comme FOUR_COLORS.
    """
    try:
        return Symbol(effect.value)
    except ValueError:
        return None

def compute_card_score(card: Card, active_symbols: list) -> int:
    # Vérification des prérequis
    pool = active_symbols.copy()
    for req in card.activation_requirements:
        if req not in pool:
            return 0
        pool.remove(req)

    # Calcul du multiplicateur
    if not card.multiplicator_effect:
        multiplier = 1


    elif multiplicator_effect.FOUR_COLORS in card.multiplicator_effect:


        multiplier = count_full_sets(

            [Symbol.RED, Symbol.BLUE, Symbol.YELLOW, Symbol.GREEN],

            active_symbols

        )


    else:
        # Convertir chaque ConditionalEffect en Symbol pour compter
        multiplier = 0
        for effect in card.multiplicator_effect:
            sym = conditional_to_symbol(effect)
            if sym is not None:
                multiplier += active_symbols.count(sym)

    return card.points * multiplier


def score_player_debug(player: Player) -> tuple:
    """
    Version debug de score_player() — affiche le détail complet
    de ce que voit le scorer pour chaque carte.
    """
    from models.card import Symbol

    print(f"\n{'='*60}")
    print(f"DEBUG SCORING — {player.name}")
    print(f"{'='*60}")

    card_scores = {}
    total = 0

    sanctuary_symbols = collect_symbols(player.sanctuaries)
    active_symbols = sanctuary_symbols.copy()

    print(f"\nSanctuaires ({len(player.sanctuaries)}) :")
    for s in player.sanctuaries:
        print(f"  [{s.id}] color={s.color!r} | symbols={[x.value for x in s.symbols]}")
    print(f"Symboles initiaux (sanctuaires) : {[x.value for x in active_symbols]}")

    print(f"\nParcours inversé des cartes :")
    for card in reversed(player.played_cards):
        # Symboles ajoutés par cette carte
        added = list(card.symbols)
        if card.color:
            try:
                added.append(Symbol(card.color))
            except ValueError:
                pass

        active_symbols.extend(card.symbols)
        if card.color:
            try:
                active_symbols.append(Symbol(card.color))
            except ValueError:
                pass

        score = compute_card_score(card, active_symbols)
        card_scores[card.id] = score
        total += score

        # Compter les symboles pertinents pour cette carte
        relevant = set(
            [Symbol(e.value) for e in card.multiplicator_effect
             if e.value in [s.value for s in Symbol]]
        ) if card.multiplicator_effect else set()

        print(f"\n  Carte [{card.id:3d}] {card.points}pts")
        print(f"    color={card.color!r} | symbols={[x.value for x in card.symbols]}")
        print(f"    reqs={[x.value for x in card.activation_requirements]}")
        print(f"    effect={[x.value for x in card.multiplicator_effect]}")
        print(f"    → ajouté au pool : {[x.value for x in added]}")
        for sym in relevant:
            count = active_symbols.count(sym)
            print(f"    → {sym.value} dans pool : {count}x")
        print(f"    → SCORE : {score}")

    print(f"\nScore cartes région : {total}")

    # Sanctuaires
    all_symbols = collect_symbols(player.played_cards + player.sanctuaries)
    sanct_total = 0
    print(f"\nScore sanctuaires :")
    for sanctuary in player.sanctuaries:
        score = compute_card_score(sanctuary, all_symbols)
        card_scores[sanctuary.id] = score
        total += score
        sanct_total += score
        print(f"  [{sanctuary.id}] {sanctuary.points}pts | "
              f"effect={[x.value for x in sanctuary.multiplicator_effect]} "
              f"→ {score}")

    print(f"\nScore sanctuaires : {sanct_total}")
    print(f"TOTAL : {total}")
    print(f"{'='*60}")

    return total, card_scores

def score_player(player: Player) -> tuple:
    card_scores = {}
    total = 0

    sanctuary_symbols = collect_symbols(player.sanctuaries)
    active_symbols = sanctuary_symbols.copy()

    for card in reversed(player.played_cards):
        active_symbols.extend(card.symbols)
        if card.color:
            try:
                active_symbols.append(Symbol(card.color))
            except ValueError:
                pass
        score = compute_card_score(card, active_symbols)
        card_scores[card.id] = score
        total += score

    all_symbols = collect_symbols(player.played_cards + player.sanctuaries)
    for sanctuary in player.sanctuaries:
        score = compute_card_score(sanctuary, all_symbols)
        card_scores[sanctuary.id] = score
        total += score

    return total, card_scores