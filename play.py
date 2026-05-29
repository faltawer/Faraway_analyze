# play.py
from __future__ import annotations
import threading
import webbrowser
import time
import random
from asyncio import wait

from ai.agents.neural_agent import NeuralAgent
from models.loader import load_cards
from models.player import Player
from models.game_state import GameState
from engine.scorer import score_player
from engine.rules import (
    get_play_order,
    draw_sanctuaries,
    discard_excess_sanctuaries,
    replenish_center,
)
from ai.agents.human_agent import HumanAgent
from ai.agents.greedy_agent import GreedyAgent
from ai.agents.probabilistic_agent import ProbabilisticAgent
from server import app, game_state_ui

CARD_FILE   = "./Documentation/faraway_data.xlsx"
CARD_FOLDER = "./Documentation/Cards"
MODEL_PATH = "./Documentation/Output/faraway_net.pt"

def setup_game(agents, card_file):
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
    state.middle_cards = [state.deck.pop(0) for _ in range(len(players) + 1)]
    for _ in range(3):
        for player in players:
            if state.deck:
                player.hand.append(state.deck.pop(0))
    return players, state


def play_round(agents, players, state):
    agent_by_name = {a.name: a for a in agents}
    gsu = game_state_ui

    # Renouvellement centre (sauf tour 1)
    if state.current_round > 1 and state.current_round != 8:
        replenish_center(state)

    # 1. Chaque joueur choisit une carte à jouer
    for agent, player in zip(agents, players):
        if isinstance(agent, HumanAgent):
            gsu["phase"] = "pick_card"
            gsu["message"] = "Choisissez une carte à jouer"
        else:
            gsu["phase"] = "ai_thinking"
        card = agent.choose_card(player, state)
        player.hand.remove(card)
        player.played_cards.append(card)
        player.current_card = card

    # 2. Ordre de jeu
    order = get_play_order(players)



    # 4. Sanctuaires (tour 2+)
    if state.current_round > 1:
        for player in order:
            draw_sanctuaries(player, state)
        for player in order:
            if player.sanctuaries_drawn:
                agent = agent_by_name[player.name]
                if len(player.sanctuaries_drawn) == 1:
                    chosen = player.sanctuaries_drawn[0]
                    player.sanctuaries.append(chosen)
                    player.sanctuaries_drawn.remove(chosen)
                else:
                    if isinstance(agent, HumanAgent):
                        gsu["phase"] = "pick_sanctuary"
                        gsu["message"] = "Choisissez un sanctuaire"
                    chosen = agent.choose_sanctuary(player, state)
                    if chosen:
                        player.sanctuaries.append(chosen)
                        player.sanctuaries_drawn.remove(chosen)
            discard_excess_sanctuaries(player, state)

    # 3. Pioche au centre (sauf dernier tour)
    if state.current_round < 8:
        for player in order:
            if state.middle_cards:
                agent = agent_by_name[player.name]
                if isinstance(agent, HumanAgent):
                    gsu["phase"] = "pick_center"
                    gsu["message"] = "Choisissez une carte dans le centre"
                pick = agent.pick_from_center(player, state)
                if pick:
                    player.hand.append(pick)
                    state.middle_cards.remove(pick)
    state.current_round += 1
    gsu["phase"] = "ai_thinking"


def run_game():
    gsu = game_state_ui
    gsu["card_folder"] = CARD_FOLDER

    agents = [
        HumanAgent("Vous"),
        # NeuralAgent("Neural", model_path=MODEL_PATH, card_file=CARD_FILE),
        GreedyAgent("Greedy 2"),
        GreedyAgent("Greedy"),
        ProbabilisticAgent("Probabiliste", card_file=CARD_FILE),
    ]

    players, state = setup_game(agents, CARD_FILE)
    gsu.update({
        "human_agent":  agents[0],
        "human_player": players[0],
        "game_state":   state,
        "all_players":  players,
        "all_agents":   agents,
        "phase":        "pick_card",
    })

    for _ in range(8):
        play_round(agents, players, state)

    scores = {p.name: score_player(p)[0] for p in players}
    gsu["scores"] = scores
    gsu["phase"]  = "finished"

    best   = max(scores.values())
    winner = [n for n, s in scores.items() if s == best]
    print(f"\n🏆 {', '.join(winner)} — {best} pts")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name}: {score} pts")


if __name__ == "__main__":
    threading.Thread(target=run_game, daemon=True).start()
    threading.Thread(
        target=lambda: (time.sleep(1.2), webbrowser.open("http://localhost:5000")),
        daemon=True
    ).start()
    print("🌐 http://localhost:5000")
    app.run(debug=False, port=5000, threaded=True)