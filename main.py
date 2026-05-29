import random
from collections import defaultdict

from ai.agents.autonomous_agent import AutonomousAgent
from ai.agents.explorationagent import ExplorationAgent
from ai.agents.learning_agent import LearningLookaheadAgent
from ai.agents.lookahead_agent import LookaheadAgent
from models import Card, Player, GameState
from ui.display import display_player_cards
from models.loader import load_cards
from engine.rules import (
    get_play_order,
    draw_sanctuaries,
    choose_sanctuary,
    discard_excess_sanctuaries,
    replenish_center, card_triggers_sanctuary,
)
from engine.scorer import score_player, score_player_debug
from ai.agents import RandomAgent, GreedyAgent, ProbabilisticAgent, DecisionTreeAgent
from ai.base_agent import Agent


def setup_game(agents: list[Agent], card_file: str) -> tuple[list[Player], GameState]:
    """
    Initialise une partie : charge les cartes, distribue les mains.
    Retourne les joueurs et le GameState initial.
    """
    region_cards, sanctuary_cards = load_cards(card_file)

    random.shuffle(region_cards)
    random.shuffle(sanctuary_cards)

    players = [Player(name=agent.name) for agent in agents]

    state = GameState(
        players=players,
        deck=region_cards,
        middle_cards=[],
        sanctuary_deck=sanctuary_cards,
        current_round=1,
    )

    # Cartes du centre : nb joueurs + 1
    state.middle_cards = [state.deck.pop(0) for _ in range(len(players) + 1)]

    # Distribution : 3 cartes en main par joueur
    for _ in range(3):
        for player in players:
            if state.deck:
                player.hand.append(state.deck.pop(0))

    return players, state


def play_round(
        agents: list[Agent],
        players: list[Player],
        state: GameState,
) -> None:
    """
    Joue un tour complet :
    1. Chaque agent choisit sa carte
    2. On détermine l'ordre de jeu
    3. On gère les sanctuaires
    4. On pioche au centre
    5. On renouvelle le centre
    """
    replenish_center(state)
    # 1. Chaque agent joue une carte
    for agent, player in zip(agents, players):
        card = agent.choose_card(player, state)
        player.hand.remove(card)
        player.played_cards.append(card)
        player.current_card = card

    # 2. Ordre de jeu (carte la plus petite en premier)
    order = get_play_order(players)
    agent_by_name = {agent.name: agent for agent in agents}

    for player in order:
        if state.current_round < 8:
            agent = agent_by_name[player.name]
            pick = agent.pick_from_center(player, state)
            player.hand.append(pick)
            state.middle_cards.remove(pick)

    # 3. Sanctuaires (seulement à partir du tour 2)
    if state.current_round > 1:
        for player in order:
            draw_sanctuaries(player, state)

        for player in order:
            if player.sanctuaries_drawn:
                agent = agent_by_name[player.name]
                chosen = agent.choose_sanctuary(player, state)
                if chosen is not None:  # ← protection
                    player.sanctuaries.append(chosen)
                    player.sanctuaries_drawn.remove(chosen)
                    discard_excess_sanctuaries(player, state)

                    # 5. Renouvellement du centre

    state.current_round += 1


def end_game(players: list[Player], card_folder: str | None = None) -> None:
    print("\n=== 🏁 Fin de partie ===")
    scores = []

    for player in players:

        if player.name == "Alice":  # ← le joueur à inspecter
            total, card_scores = score_player_debug(player)
        else:
            total, card_scores = score_player(player)
        # total, card_scores = score_player(player)
        scores.append((player.name, total))
        print(f"  {player.name} : {total} pts")

        # Génération de l'image si card_folder fourni
        if card_folder:
            display_player_cards(
                played_card_ids=[c.id for c in player.played_cards],
                sanctuary_ids=[c.id for c in player.sanctuaries],
                card_folder=card_folder,
                score_dict=card_scores,
                output_path=f"./Documentation/Output/faraway_{player.name}.jpg",
            )

    best = max(scores, key=lambda x: x[1])[1]
    winners = [name for name, score in scores if score == best]
    if len(winners) == 1:
        print(f"\n🏆 Gagnant : {winners[0]} avec {best} pts !")
    else:
        print(f"\n🤝 Égalité : {', '.join(winners)} avec {best} pts !")


def run_learning_selfplay(
        n: int,
        card_file: str,
        depth: int = 3,
        save_path: str = "q_table.pkl",
) -> None:
    agents = [
        LearningLookaheadAgent("Learner_1", depth=depth),
        LearningLookaheadAgent("Learner_2", depth=depth),
        LearningLookaheadAgent("Learner_3", depth=depth),
        LearningLookaheadAgent("Learner_4", depth=depth),
    ]

    totals = defaultdict(float)
    wins = defaultdict(int)
    max_scores = defaultdict(int)
    # Suivi de la progression
    history_avg = []

    for i in range(n):
        players, state = setup_game(agents, card_file)
        for _ in range(8):
            play_round(agents, players, state)

        # Scores finaux
        scores = {}
        for player in players:
            total, _ = score_player(player)
            scores[player.name] = total
            totals[player.name] += total
            max_scores[player.name] = max(max_scores[player.name], total)

        all_scores = list(scores.values())

        # Apprentissage — propager la récompense sur toute la partie
        for agent, player in zip(agents, players):
            agent.learn_from_game(scores[player.name], all_scores)

        # Victoire
        winner = max(scores, key=scores.get)
        wins[winner] += 1

        # Log tous les 50 tours
        if (i + 1) % 50 == 0:
            avg = sum(totals.values()) / (len(agents) * (i + 1))
            eps = agents[0].epsilon
            history_avg.append(avg)
            print(f"  [{i + 1:4d}/{n}] score moyen={avg:.1f} | "
                  f"epsilon={eps:.3f}")

    print(f"\n=== Self-play — {n} parties | depth={depth} ===")
    for agent in agents:
        name = agent.name
        avg = totals[name] / n
        w = wins[name]
        print(f"  {name}: score moyen={avg:5.1f} | "
              f"max={max_scores[name]:3d} | "
              f"victoires={w:3d}/{n} ({100 * w / n:.0f}%)")

    # Tendance : est-ce que les scores augmentent ?
    if len(history_avg) >= 2:
        trend = history_avg[-1] - history_avg[0]
        print(f"\n📈 Progression du score moyen : "
              f"{history_avg[0]:.1f} → {history_avg[-1]:.1f} "
              f"({'+' if trend > 0 else ''}{trend:.1f})")

    agents[0].save_q(save_path)


def run_selfplay(
        n: int,
        card_file: str,
        depth: int = 3,
        n_simulations: int = 20,
) -> None:
    """
    Lance n parties Lookahead vs Lookahead.
    Enregistre les scores moyens pour valider la profondeur optimale.
    """
    from collections import defaultdict
    totals = defaultdict(float)
    wins = defaultdict(int)
    max_scores = defaultdict(int)

    for i in range(n):
        agents = [
            LookaheadAgent("Lookahead_1", depth=depth, n_simulations=n_simulations),
            LookaheadAgent("Lookahead_2", depth=depth, n_simulations=n_simulations),
            RandomAgent("Random_1"),
            GreedyAgent("Greedy_1"),
        ]
        players, state = setup_game(agents, card_file)
        for _ in range(8):
            play_round(agents, players, state)

        scores = {}
        for player in players:
            total, _ = score_player(player)
            scores[player.name] = total
            totals[player.name] += total
            # Mise à jour du score max pour cet agent
            if total > max_scores[player.name]:
                max_scores[player.name] = total

        winner = max(scores, key=scores.get)
        wins[winner] += 1

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n} parties simulées...")

    print(f"\n=== Self-play — {n} parties | depth={depth} ===")
    for name in totals.keys():  # ou wins.keys()
        avg = totals[name] / n
        max_score = max_scores[name]  # Score max de cet agent sur toutes les simulations
        w = wins[name]
        print(f"  {name}: score moyen={avg:.1f} | max={max_score:.1f} | victoires={w}/{n} ({100 * w / n:.0f}%)")
    end_game(players, card_folder=CARD_FOLDER)


# Dans run_game(), passer card_folder :
def run_game(agents: list[Agent], card_file: str, card_folder: str | None = None) -> None:
    players, state = setup_game(agents, card_file)
    for round_num in range(1, 9):
        play_round(agents, players, state)

    end_game(players, card_folder=card_folder)


def run_simulations(n: int, card_file: str) -> None:
    """Lance n parties et compare les scores moyens par type d'agent."""
    from collections import defaultdict
    totals = defaultdict(int)
    wins = defaultdict(int)
    max_scores = defaultdict(int)

    for i in range(n):
        agents = [
            RandomAgent("Random_1"),
            RandomAgent("Random_2"),
            GreedyAgent("Greedy_1"),
            GreedyAgent("Greedy_2"),
        ]
        players, state = setup_game(agents, card_file)
        for round_num in range(1, 9):
            play_round(agents, players, state)

        scores = {}
        for player in players:
            total, _ = score_player(player)
            scores[player.name] = total
            totals[player.name] += total
            # Mise à jour du score max pour cet agent
            if total > max_scores[player.name]:
                max_scores[player.name] = total

        winner = max(scores, key=scores.get)
        wins[winner] += 1

    print(f"\n=== Résultats sur {n} simulations ===")
    for name in totals.keys():  # ou wins.keys()
        avg = totals[name] / n
        max_score = max_scores[name]  # Score max de cet agent sur toutes les simulations
        w = wins[name]
        print(f"  {name}: score moyen={avg:.1f} | max={max_score:.1f} | victoires={w}/{n} ({100 * w / n:.0f}%)")


def run_trained_agent(card_file: str, q_table_path: str):
    agent = LearningLookaheadAgent("Trained_Agent", depth=3)
    agent.load_q(q_table_path)
    agents = [
        agent,
        RandomAgent("Random"),
        GreedyAgent("Greedy"),

    ]
    run_game(agents, card_file, card_folder=CARD_FOLDER)

def find_theoretical_max(card_file: str, n: int = 1000) -> None:
    """
    Joue n parties avec un agent omniscient (voit toutes les cartes)
    pour trouver le score maximum atteignable.
    """
    from models.loader import load_cards
    from engine.scorer import score_player

    max_score = 0
    scores = []

    for _ in range(n):
        region_cards, sanctuary_cards = load_cards(card_file)
        random.shuffle(region_cards)
        random.shuffle(sanctuary_cards)

        # Donner les 8 meilleures cartes possible à un joueur
        player = Player("Oracle")
        player.played_cards = region_cards[:8]
        player.sanctuaries = sanctuary_cards[:4]

        score, _ = score_player(player)
        scores.append(score)
        max_score = max(max_score, score)

    import statistics
    print(f"\n=== Plafond théorique ({n} tirages aléatoires) ===")
    print(f"  Score max absolu  : {max_score}")
    print(f"  Moyenne           : {statistics.mean(scores):.1f}")
    print(f"  Médiane           : {statistics.median(scores):.1f}")
    print(f"  Écart-type        : {statistics.stdev(scores):.1f}")

    def run_autonomous_training(
            n: int,
            card_file: str,
            save_path: str = "autonomous.pkl",
    ) -> None:
        """
        Laisse l'IA apprendre seule.
        Aucune stratégie injectée — juste le scoring et la récompense relative.
        """
        agents = [
            AutonomousAgent("Auto_1", n_simulations=50),
            AutonomousAgent("Auto_2", n_simulations=50),
            AutonomousAgent("Auto_3", n_simulations=50),
            AutonomousAgent("Auto_4", n_simulations=50),
        ]

        totals = defaultdict(float)
        wins = defaultdict(int)
        checkpoints = []  # score moyen tous les 100 tours

        for i in range(n):
            players, state = setup_game(agents, card_file)
            for _ in range(8):
                play_round(agents, players, state)

            scores = {}
            for player in players:
                total, _ = score_player(player)
                scores[player.name] = total
                totals[player.name] += total

            all_scores = list(scores.values())

            # Apprentissage autonome
            for agent, player in zip(agents, players):
                agent.learn_from_game(scores[player.name], all_scores)

            winner = max(scores, key=scores.get)
            wins[winner] += 1

            # Checkpoint tous les 100 tours
            if (i + 1) % 100 == 0:
                avg = sum(totals.values()) / (len(agents) * (i + 1))
                checkpoints.append((i + 1, avg))
                print(f"  [{i + 1:5d}/{n}] score moyen = {avg:.1f}")

        # Courbe de progression
        print("\n=== Progression de l'apprentissage ===")
        for step, avg in checkpoints:
            bar = "█" * int(avg / 2)
            print(f"  {step:5d} parties : {avg:5.1f} pts {bar}")

        print(f"\n=== Résultats finaux ===")
        for agent in agents:
            avg = totals[agent.name] / n
            w = wins[agent.name]
            print(f"  {agent.name}: score moyen={avg:.1f} | "
                  f"victoires={w}/{n} ({100 * w / n:.0f}%)")

        agents[0].save(save_path)


# ==============================
def run_autonomous_training(
        n: int,
        card_file: str,
        save_path: str = "autonomous.pkl",
) -> None:
    agents = [
        AutonomousAgent("Auto_1", n_simulations=50),
        AutonomousAgent("Auto_2", n_simulations=50),
        AutonomousAgent("Auto_3", n_simulations=50),
        AutonomousAgent("Auto_4", n_simulations=50),
    ]

    totals = defaultdict(float)
    wins = defaultdict(int)
    checkpoints = []
    last_players = None  # ← garder la dernière partie pour affichage

    for i in range(n):
        players, state = setup_game(agents, card_file)
        for _ in range(8):
            play_round(agents, players, state)

        # Calcul des scores — UNE SEULE FOIS par partie
        scores = {}
        for player in players:
            total, _ = score_player(player)
            scores[player.name] = total
            totals[player.name] += total

        all_scores = list(scores.values())

        # Apprentissage
        for agent, player in zip(agents, players):
            agent.learn_from_game(scores[player.name], all_scores)

        winner = max(scores, key=scores.get)
        wins[winner] += 1
        last_players = players  # ← mémoriser pour affichage final

        # Checkpoint tous les 100 tours
        if (i + 1) % 100 == 0:
            avg = sum(totals.values()) / (len(agents) * (i + 1))
            checkpoints.append((i + 1, avg))
            print(f"  [{i + 1:5d}/{n}] score moyen = {avg:.1f}")

    # Résultats
    print("\n=== Progression de l'apprentissage ===")
    for step, avg in checkpoints:
        bar = "█" * int(avg / 2)
        print(f"  {step:5d} parties : {avg:5.1f} pts {bar}")

    print(f"\n=== Résultats finaux ===")
    for agent in agents:
        avg = totals[agent.name] / n
        w = wins[agent.name]
        print(f"  {agent.name}: score moyen={avg:.1f} | "
              f"victoires={w}/{n} ({100 * w / n:.0f}%)")

    agents[0].save(save_path)
    end_game(players, card_folder=CARD_FOLDER)


def run_comparison(n: int, card_file: str, agents: None) -> None:
    from collections import defaultdict

    totals = defaultdict(float)
    wins = defaultdict(int)

    for i in range(n):
        players, state = setup_game(agents, card_file)
        for _ in range(8):
            play_round(agents, players, state)

        scores = {}
        for player in players:
            total, _ = score_player(player)
            scores[player.name] = total
            totals[player.name] += total

        winner = max(scores, key=scores.get)
        wins[winner] += 1

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{n}...")

    print(f"\n=== Comparaison sur {n} parties ===")
    print(f"  {'Agent':<15} | {'Moy':>6} | {'Min':>5} | {'Victoires':>10}")
    print(f"  {'-' * 15}-+-{'-' * 6}-+-{'-' * 5}-+-{'-' * 10}")
    for name in ["Random", "Greedy", "Probabiliste", "Autonomous"]:
        avg = totals[name] / n
        w = wins[name]
        bar = "█" * int(avg / 2)
        print(f"  {name:<15} | {avg:>6.1f} | {w:>3}/{n} ({100 * w / n:>3.0f}%) | {bar}")


def play_game(agents, card_file):
    players, state = setup_game(agents, card_file)
    for agent in agents:
        if hasattr(agent, "start_new_game"):
            agent.start_new_game(players[agents.index(agent)], state)

    for round_num in range(1, 9):
        play_round(agents, players, state)

    scores = {}
    for player in players:
        total, _ = score_player(player)
        scores[player.name] = total
        for agent in agents:
            if agent.name == player.name and hasattr(agent, "learn_from_game"):
                agent.learn_from_game(total, player, state)
    return scores


if __name__ == "__main__":
    CARD_FILE = "./Documentation/faraway_data.xlsx"
    CARD_FOLDER = r"R:\FLo\GitHub\Faraway-Project\Documentation\Cards"
    MODEL_PATH = "./Documentation/Output/faraway_net.pt"
    DATA_PATH = "./Documentation/Output/training_data.npz"
    # find_theoretical_max(CARD_FILE, n=10000)
    run_autonomous_training(100000, CARD_FILE, save_path="autonomous.pkl")
    # run_simulations(1000, CARD_FILE)
    # #
    # agent = AutonomousAgent("Auto", card_file=CARD_FILE)
    # #
    # agents = [
    #     # ExplorationAgent("Explorer_1", exploration_rate=0.7),
    #     GreedyAgent("Greedy"),
    #     GreedyAgent("Greedy"),
    #     GreedyAgent("Greedy"),
    # ]

    # Lancer 1000 parties pour explorer et enregistrer des stratégies
    # for _ in range(10000):
    #     play_game(agents, CARD_FILE)
    #     if _ % 1000 == 0:
    #         print(f"{_}Game")
    # # Après entraînement — sauvegarder
    # agents = [
    #     RandomAgent("Random"),
    #     GreedyAgent("Greedy"),
    #     ProbabilisticAgent("Probabiliste", card_file=CARD_FILE),
    #     AutonomousAgent("Autonomous", card_file=CARD_FILE, load_path="autonomous.pkl"),
    # ]
    # run_comparison(100, CARD_FILE, agents)
    # run_game(agents, CARD_FILE, card_folder=CARD_FOLDER)
    # Exemple d'entraînement (100 parties)
    # run_learning_selfplay(500, CARD_FILE, depth=3, save_path="q_table.pkl")
    # run_trained_agent(CARD_FILE, "q_table.pkl")
    # run_selfplay(1, CARD_FILE, depth=8, n_simulations=40)
