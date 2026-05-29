# ai/neural/train.py
from __future__ import annotations
import os
import random
import numpy as np
from collections import defaultdict

from ai.agents.AggressiveAgent import AggressiveAgent

CARD_FILE   = "./Documentation/faraway_data.xlsx"
MODEL_PATH  = "./Documentation/Output/faraway_net.pt"
DATA_PATH   = "./Documentation/Output/training_data.npz"
N_COLLECT   = 2000
N_EVAL      = 100
EPOCHS      = 80
WEIGHT_MODE = "exponential"  # "linear" | "exponential" | "step"

from models.loader import load_cards
from models.player import Player
from models.game_state import GameState
from engine.scorer import score_player
from engine.rules import (
    get_play_order, draw_sanctuaries,
    discard_excess_sanctuaries, replenish_center,
)
from ai.agents.random_agent import RandomAgent
from ai.agents.greedy_agent import GreedyAgent
from ai.agents.probabilistic_agent import ProbabilisticAgent
from ai.agents.neural_agent import NeuralAgent
from ai.neural.model import FarawayNet, FarawayTrainer
from ai.neural.data_collector import DataCollector


def setup_game(agents, card_file):
    region_cards, sanctuary_cards = load_cards(card_file)
    random.shuffle(region_cards)
    random.shuffle(sanctuary_cards)
    players = [Player(name=a.name) for a in agents]
    state = GameState(
        players=players, deck=region_cards,
        middle_cards=[], sanctuary_deck=sanctuary_cards,
        current_round=1,
    )
    state.middle_cards = [state.deck.pop(0) for _ in range(len(players) + 1)]
    for _ in range(3):
        for p in players:
            if state.deck:
                p.hand.append(state.deck.pop(0))
    return players, state


def play_round(agents, players, state):
    if state.current_round > 1:
        replenish_center(state)
    agent_by_name = {a.name: a for a in agents}

    for agent, player in zip(agents, players):
        card = agent.choose_card(player, state)
        player.hand.remove(card)
        player.played_cards.append(card)
        player.current_card = card

    order = get_play_order(players)

    if state.current_round < 8:
        for player in order:
            if state.middle_cards:
                pick = agent_by_name[player.name].pick_from_center(player, state)
                if pick:
                    player.hand.append(pick)
                    state.middle_cards.remove(pick)

    if state.current_round > 1:
        for player in order:
            draw_sanctuaries(player, state)
        for player in order:
            if player.sanctuaries_drawn:
                chosen = agent_by_name[player.name].choose_sanctuary(player, state)
                if chosen:
                    player.sanctuaries.append(chosen)
                    player.sanctuaries_drawn.remove(chosen)
            discard_excess_sanctuaries(player, state)

    state.current_round += 1


# ------------------------------------------------------------------ #
#  Collecte                                                           #
# ------------------------------------------------------------------ #

def collect_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retourne X (features), y (scores finaux), w (poids par tour).

    Poids selon WEIGHT_MODE :
      Tour 1 → 1.0×   Tour 8 → 4.0×  (exponential)
    """
    print(f"\n{'='*55}")
    print(f"COLLECTE — {N_COLLECT} parties | poids : {WEIGHT_MODE}")
    print(f"{'='*55}")

    collector = DataCollector(CARD_FILE)

    for i in range(N_COLLECT):
        r = random.random()
        if r < 0.50:
            def factory():
                return [
                    GreedyAgent("target"),
                    GreedyAgent("G2"),
                    RandomAgent("R1"),
                    ProbabilisticAgent("P1", CARD_FILE),
                ]
        elif r < 0.85:
            def factory():
                return [
                    ProbabilisticAgent("target", CARD_FILE),
                    GreedyAgent("G1"),
                    GreedyAgent("G2"),
                    RandomAgent("R1"),
                ]
        else:
            # Seulement 15% de random pour la diversité minimale
            def factory():
                return [
                    NeuralAgent("target", model_path=MODEL_PATH, card_file=CARD_FILE, epsilon=0.2),
                    NeuralAgent("N2", model_path=MODEL_PATH, card_file=CARD_FILE, epsilon=0.2),
                    NeuralAgent("N3", model_path=MODEL_PATH, card_file=CARD_FILE, epsilon=0.2),
                    ProbabilisticAgent("P1", CARD_FILE),
                ]
        # collector._run_game(factory, "target")

        collector._run_game(factory, "target")

        if (i + 1) % 200 == 0:
            print(f"  {i+1:5d}/{N_COLLECT} | {len(collector.X):6d} exemples")

    X = np.stack(collector.X)
    y = np.array(collector.y,       dtype=np.float32)
    w = np.array(collector.weights, dtype=np.float32)

    # Stats de vérification
    n = len(w)
    w_early = w[:n // 3].mean()
    w_late  = w[2 * n // 3:].mean()
    print(f"\n✅ {n} exemples collectés")
    print(f"   Score moyen     : {y.mean():.1f} ± {y.std():.1f}")
    print(f"   Min / Max       : {y.min():.0f} / {y.max():.0f}")
    print(f"   Poids tours 1-3 : {w_early:.2f}")
    print(f"   Poids tours 6-8 : {w_late:.2f}  (×{w_late/w_early:.1f})")

    return X, y, w


# ------------------------------------------------------------------ #
#  Entraînement                                                       #
# ------------------------------------------------------------------ #

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray | None = None,
) -> FarawayNet:
    print(f"\n{'='*55}")
    print(f"ENTRAÎNEMENT — {EPOCHS} époques | {len(X)} exemples")
    print(f"  Loss pondérée : {'oui' if w is not None else 'non'}")
    print(f"{'='*55}")

    model   = FarawayNet(hidden_dims=(128, 64, 32))
    trainer = FarawayTrainer(model, lr=1e-3, batch_size=64)

    # Normaliser y ∈ [0, 1]
    y_norm = y / 100.0

    # Normaliser w autour de 1.0 — évite d'exploser la loss
    w_norm = None
    if w is not None:
        w_norm = w / w.mean()

    trainer.train(
        X, y_norm,
        weights   = w_norm,
        epochs    = EPOCHS,
        val_split = 0.1,
        verbose   = True,
    )

    print(f"\n✅ Entraînement terminé")
    # print(f"   Perte finale (train) : {trainer.losses[-1]:.4f}")
    return model


# ------------------------------------------------------------------ #
#  Évaluation                                                         #
# ------------------------------------------------------------------ #

def evaluate(model_path: str) -> None:
    print(f"\n{'='*55}")
    print(f"ÉVALUATION — {N_EVAL} parties")
    print(f"{'='*55}")

    totals = defaultdict(float)
    wins   = defaultdict(int)

    for i in range(N_EVAL):
        agents = [
            NeuralAgent("Neural", model_path=model_path, card_file=CARD_FILE),
            ProbabilisticAgent("Probabiliste", CARD_FILE),
            GreedyAgent("Greedy"),
            RandomAgent("Random"),
        ]
        players, state = setup_game(agents, CARD_FILE)
        for _ in range(8):
            play_round(agents, players, state)

        scores = {p.name: score_player(p)[0] for p in players}
        for name, sc in scores.items():
            totals[name] += sc
        winner = max(scores, key=scores.get)
        wins[winner] += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{N_EVAL}...")

    print(f"\n  {'Agent':<15} | {'Moy':>6} | Victoires")
    print(f"  {'-'*15}-+-{'-'*6}-+-{'-'*22}")
    for name in ["Neural", "Probabiliste", "Greedy", "Random"]:
        avg = totals[name] / N_EVAL
        w_  = wins[name]
        bar = "█" * int(avg / 2)
        print(f"  {name:<15} | {avg:>6.1f} | "
              f"{w_:>3}/{N_EVAL} ({100*w_/N_EVAL:>3.0f}%) {bar}")

def iterative_training(
    n_rounds: int = 3,
    n_collect_per_round: int = 1000,
) -> None:
    """
    Entraînement itératif :
      Round 1 : apprend depuis Greedy + Proba
      Round 2 : apprend depuis lui-même (v1) + Proba
      Round 3 : apprend depuis lui-même (v2) + Proba
      ...

    À chaque round, le réseau devient son propre professeur.
    """
    global N_COLLECT
    N_COLLECT = n_collect_per_round

    for round_num in range(1, n_rounds + 1):
        print(f"\n{'#'*55}")
        print(f"  ITÉRATION {round_num}/{n_rounds}")
        print(f"{'#'*55}")

        # Utiliser le meilleur modèle comme agent cible
        if round_num == 1 or not os.path.exists(MODEL_PATH):
            # Première itération : données Greedy + Proba
            target_factory = None
        else:
            # Itérations suivantes : le réseau joue contre lui-même
            def make_neural():
                return NeuralAgent(
                    "target",
                    model_path=MODEL_PATH,
                    card_file=CARD_FILE,
                    epsilon=0.15,  # plus d'exploration en self-play
                )

        collector = DataCollector(CARD_FILE)

        for i in range(N_COLLECT):
            r = random.random()
            if r < 0.50:
                def factory():
                    return [
                        GreedyAgent("target"),
                        GreedyAgent("G2"),
                        RandomAgent("R1"),
                        ProbabilisticAgent("P1", CARD_FILE),
                    ]
            elif r < 0.85:
                def factory():
                    return [
                        ProbabilisticAgent("target", CARD_FILE),
                        GreedyAgent("G1"),
                        GreedyAgent("G2"),
                        RandomAgent("R1"),
                    ]
            else:
                # Seulement 15% de random pour la diversité minimale
                def factory():
                    return [
            NeuralAgent("target", model_path=MODEL_PATH, card_file=CARD_FILE, epsilon=0.2),
            NeuralAgent("N2", model_path=MODEL_PATH, card_file=CARD_FILE, epsilon=0.2),
            NeuralAgent("N3", model_path=MODEL_PATH, card_file=CARD_FILE, epsilon=0.2),
            ProbabilisticAgent("P1", CARD_FILE),
        ]
            collector._run_game(factory, "target")

        X = np.stack(collector.X)
        y = np.array(collector.y, dtype=np.float32)
        w = np.array(collector.weights, dtype=np.float32)

        threshold = np.percentile(y, 80)  # top 20%
        mask = y >= threshold

        print(f"  Score moyen collecté : {y.mean():.1f}")
        X = X[mask]
        y = y[mask]
        w = w[mask]
        print(f"  Nouveau score moyen : {y.mean():.1f}")
        # Entraîner sur ces nouvelles données
        model = train_model(X, y, w)
        model.save(MODEL_PATH)

        # Évaluer
        evaluate(MODEL_PATH)
# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    os.makedirs("./Documentation/Output", exist_ok=True)

    # os.makedirs("./Documentation/Output", exist_ok=True)
    # Lancer l'entraînement itératif
    # iterative_training(n_rounds=3, n_collect_per_round=1000)
    # ── 1. Données ──────────────────────────────────────────────────
    # if os.path.exists(DATA_PATH):
    #     print(f"📂 Données existantes : {DATA_PATH}")
    #     choice = input("  Recollectionner ? (o/N) ").strip().lower()
    #     if choice == "o":
    #         X, y, w = collect_data()
    #         np.savez_compressed(DATA_PATH, X=X, y=y, w=w)
    #         print(f"💾 Sauvegardé : {DATA_PATH}")
    #     else:
    #         data = np.load(DATA_PATH)
    #         X    = data["X"]
    #         y    = data["y"]
    #         w    = data["w"] if "w" in data else None
    #         print(f"✅ {len(X)} exemples chargés"
    #               + (" (sans poids — recollectionner pour activer)" if w is None else ""))
    # else:
    X, y, w = collect_data()
    np.savez_compressed(DATA_PATH, X=X, y=y, w=w)
    print(f"💾 Sauvegardé : {DATA_PATH}")

    # ── 2. Entraînement ─────────────────────────────────────────────
    model = train_model(X, y, w)
    model.save(MODEL_PATH)

    # ── 3. Évaluation ───────────────────────────────────────────────
    evaluate(MODEL_PATH)