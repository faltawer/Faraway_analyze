# server.py
from __future__ import annotations
import os
from flask import Flask, render_template, jsonify, request, send_file
from models.card import Card
from models.player import Player
from engine.scorer import score_player

app = Flask(__name__)

game_state_ui = {
    "phase": "waiting",
    # waiting | pick_card | pick_center | pick_sanctuary | ai_thinking | finished
    "human_player": None,
    "game_state":   None,
    "human_agent":  None,
    "all_players":  [],
    "all_agents":   [],
    "message":      "",
    "scores":       {},
    "card_folder":  "",
}


def card_to_dict(card: Card, score: int | None = None) -> dict:
    return {
        "id":           card.id,
        "points":       card.points,
        "color":        card.color,
        "symbols":      [s.value for s in card.symbols],
        "requirements": [r.value for r in card.activation_requirements],
        "effect":       [e.value for e in card.multiplicator_effect],
        "score":        score,
        "img_url":      f"/card_image/{card.id}",
    }


def player_to_dict(player: Player, show_score: bool = False) -> dict:
    score_dict = {}
    total = 0
    if show_score:
        total, score_dict = score_player(player)
    return {
        "name":         player.name,
        "played_cards": [card_to_dict(c, score_dict.get(c.id)) for c in player.played_cards],
        "sanctuaries":  [card_to_dict(c, score_dict.get(c.id)) for c in player.sanctuaries],
        "hand_count":   len(player.hand),
        "total_score":  total,
    }


@app.route("/")
def index():
    return render_template("game.html")


@app.route("/card_image/<int:card_id>")
def card_image(card_id: int):
    folder = game_state_ui["card_folder"]
    path = os.path.join(folder, f"card_{card_id}.jpg")
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "", 404


@app.route("/api/state")
def api_state():
    gsu   = game_state_ui
    human = gsu["human_player"]
    state = gsu["game_state"]

    if not human or not state:
        return jsonify({"phase": "waiting"})

    finished = gsu["phase"] == "finished"
    current_score, _ = score_player(human)

    return jsonify({
        "phase":             gsu["phase"],
        "message":           gsu["message"],
        "round":             state.current_round,
        "hand":              [card_to_dict(c) for c in human.hand],
        "center":            [card_to_dict(c) for c in state.middle_cards],
        "sanctuaries_drawn": [card_to_dict(c) for c in human.sanctuaries_drawn],
        "players":           [player_to_dict(p, show_score=finished) for p in gsu["all_players"]],
        "human_name":        human.name,
        "current_score":     current_score,
        "deck_count":        len(state.deck),
        "scores":            gsu["scores"],
    })


@app.route("/api/play_card/<int:card_id>", methods=["POST"])
def play_card(card_id: int):
    gsu = game_state_ui
    if gsu["phase"] != "pick_card":
        return jsonify({"error": "Pas le bon moment"}), 400
    human = gsu["human_player"]
    card  = next((c for c in human.hand if c.id == card_id), None)
    if not card:
        return jsonify({"error": "Carte introuvable"}), 404
    gsu["human_agent"].set_card_choice(card)
    return jsonify({"ok": True})


@app.route("/api/pick_center/<int:card_id>", methods=["POST"])
def pick_center(card_id: int):
    gsu   = game_state_ui
    if gsu["phase"] != "pick_center":
        return jsonify({"error": "Pas le bon moment"}), 400
    state = gsu["game_state"]
    card  = next((c for c in state.middle_cards if c.id == card_id), None)
    if not card:
        return jsonify({"error": "Carte introuvable"}), 404
    gsu["human_agent"].set_center_choice(card)
    return jsonify({"ok": True})


@app.route("/api/pick_sanctuary/<int:card_id>", methods=["POST"])
def pick_sanctuary(card_id: int):
    gsu   = game_state_ui
    if gsu["phase"] != "pick_sanctuary":
        return jsonify({"error": "Pas le bon moment"}), 400
    human = gsu["human_player"]
    card  = next((c for c in human.sanctuaries_drawn if c.id == card_id), None)
    if not card:
        return jsonify({"error": "Sanctuaire introuvable"}), 404
    gsu["human_agent"].set_sanctuary_choice(card)
    return jsonify({"ok": True})