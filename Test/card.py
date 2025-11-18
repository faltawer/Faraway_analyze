import os
import pandas as pd
import random
from PIL import Image, ImageDraw, ImageFont

card_path = r"/Documentation/Cards"


class Card:
    """Represents a Faraway card with various gameplay attributes."""

    def __init__(self, id, points, color, symbols, conditional_effect, activation_requirements, type):
        self.id = id
        self.points = points
        self.color = color
        self.type = type
        self.symbols = symbols
        self.conditional_effect = conditional_effect
        self.activation_requirements = activation_requirements

    def __repr__(self):
        return f"<Card {self.id} - Symbols: {self.symbols} - Condition: {self.conditional_effect} - Requirements: {self.activation_requirements} - {self.points} pts - Color: {self.color}>"


class Player:
    """Represents a player in the game with their hand, played cards, and sanctuaries."""

    def __init__(self, name):
        self.name = name
        self.hand = []
        self.played_cards = []
        self.sanctuaries = []
        self.current_card = None
        self.sanctuaries_drawn = []

    def draw_card(self, card):
        """Adds a card to the player's hand."""
        self.hand.append(card)

    def extract_ids(cards):
        """Utility method to extract card IDs from a list of Card objects."""
        return [card.id for card in cards]

    def count_full_sets(self, reqs, pool):
        """
        Returns how many complete sets of all required elements can be formed from the pool.
        Useful for cards requiring combinations like ['r', 'g', 'b', 'y'].
        """
        if not reqs:
            return 0
        return min(pool.count(r) // reqs.count(r) for r in set(reqs))

    def calculate_score(self):
        """Computes total score for the player based on their played cards and sanctuaries."""
        active_symbols = []
        active_colors = []
        card_scores = {}

        # Initial active symbols: sanctuaries
        for s in self.sanctuaries:
            active_symbols.extend(s.symbols)
            if pd.notna(s.color):
                active_symbols.append(s.color)

        total_score = 0
        for card in self.played_cards[::-1]:
            active_symbols.append(card.color)
            active_symbols.extend(card.symbols)

            reqs_met = True
            active_symbols_copy = active_symbols.copy()
            for req in card.activation_requirements:
                if req not in active_symbols_copy:
                    reqs_met = False
                    break
                else:
                    active_symbols_copy.remove(req)

            if reqs_met:
                if card.conditional_effect == ['4colors']:
                    multiplier = self.count_full_sets(['r', 'b', 'y', 'g'], active_symbols)
                elif card.conditional_effect:
                    multiplier = sum(active_symbols.count(sym) for sym in card.conditional_effect)
                else:
                    multiplier = 1
                score = card.points * multiplier
                total_score += score
                card_scores[card.id] = score
            else:
                card_scores[card.id] = 0

        # Sanctuary score
        add_sanctuaries = 0
        for s in self.sanctuaries:
            if s.conditional_effect == ['4colors']:
                multiplier = self.count_full_sets(['r', 'b', 'y', 'g'], active_symbols)
            elif s.conditional_effect:
                multiplier = sum(active_symbols.count(sym) for sym in s.conditional_effect)
            else:
                multiplier = 1
            score = multiplier * s.points
            add_sanctuaries += score
            card_scores[s.id] = score

        return total_score + add_sanctuaries, card_scores

    def choose_card_intelligent(self):
        """Placeholder logic: randomly chooses a card to play from hand."""
        self.current_card = random.choice(self.hand)
        self.hand.remove(self.current_card)
        self.played_cards.append(self.current_card)

    def count_clues(self):
        """Counts how many 'map' symbols are present in the played + sanctuary cards."""
        return sum(c.symbols.count('map') for c in self.played_cards + self.sanctuaries)

    def pick_from_center(self, center):
        """Chooses a random card from the center pool."""
        pick = random.choice(center)
        self.hand.append(pick)
        center.remove(pick)

    def choose_sanctuary(self):
        """Chooses one sanctuary from the drawn options."""
        if self.sanctuaries_drawn:
            chosen = random.choice(self.sanctuaries_drawn)
            self.sanctuaries.append(chosen)
            self.sanctuaries_drawn = []

    def discard_excess_sanctuaries(self, sanctuary_deck):
        """Ensures the player has at most as many sanctuaries as played cards."""
        while self.sanctuaries_drawn:
            sanctuary_deck.append(self.sanctuaries_drawn.pop())

        while len(self.sanctuaries) > len(self.played_cards):
            sanctuary_deck.append(self.sanctuaries.pop())

    def card_activatable(self, card):
        """Checks whether the given card can be activated based on owned symbols."""
        owned = [sym for c in self.played_cards + self.sanctuaries for sym in c.symbols]
        return all(req in owned for req in card.activation_requirements)

    def __repr__(self):
        return f"Player {self.name} (Hand: {len(self.hand)} cards, Played: {len(self.played_cards)}, Sanctuaries: {len(self.sanctuaries)})"


class Game:
    """Core game logic for Faraway."""

    def __init__(self, player_names, card_file):
        self.player_names = player_names
        self.card_file = card_file
        self.players = []
        self.deck = []
        self.middle_cards = []
        self.sanctuary_deck = []
        self.check_numbers_players()

    def check_numbers_players(self):
        """Raises error if number of players is outside the 2–6 range."""
        if len(self.player_names) > 6:
            raise ValueError("Maximum number of players is 6.")
        if len(self.player_names) < 2:
            raise ValueError("Minimum number of players is 2.")

    def safe_split(self, value):
        """Safely splits a comma-separated string into a list."""
        if pd.isna(value):
            return []
        return [x.strip() for x in str(value).split(',')]

    def display_faraway_cards(self, played_cards, sanctuaries, card_folder, score_dict=None, output_path="output.jpg"):
        """
        Generates a visual image of all cards played and sanctuaries for a player,
        adding score labels at bottom left.
        """
        output_path = f"./Documentation/{output_path}"

        def load_card_image(card_id):
            path = os.path.join(card_folder, f"card_{card_id}.jpg")
            return Image.open(path)

        if not played_cards:
            raise ValueError("played_cards list cannot be empty.")

        sample_img = load_card_image(played_cards[0])
        card_width, card_height = sample_img.size

        cols = 4
        rows = 2
        padding = 10
        num_sanctuaries = len(sanctuaries)

        total_width = cols * card_width + (cols + 1) * padding
        sanctuary_height = card_height if num_sanctuaries > 0 else 0
        total_height = sanctuary_height + (rows * card_height) + (rows + 2) * padding

        canvas = Image.new('RGB', (total_width, total_height), 'white')

        # Paste sanctuaries
        for idx, card_id in enumerate(sanctuaries):
            img = load_card_image(card_id)
            x = padding + idx * (card_width + padding)
            y = padding
            score = score_dict.get(card_id, None) if score_dict else None
            if score is not None:
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 28)
                except:
                    font = ImageFont.load_default()
                score_text = f"{score} pts"
                text_width, text_height = draw.textsize(score_text, font=font)
                draw.rectangle([5, img.height - text_height - 10, 20 + text_width, img.height - 8], fill="white")
                draw.text((10, img.height - text_height - 10), score_text, fill="black", font=font)
            canvas.paste(img, (x, y))

        # Paste played cards
        for i, card_id in enumerate(played_cards):
            img = load_card_image(card_id)
            row = i // cols
            col = i % cols
            x = padding + col * (card_width + padding)
            y = sanctuary_height + (2 * padding) + row * (card_height + padding)
            score = score_dict.get(card_id, None) if score_dict else None
            if score is not None:
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 28)
                except:
                    font = ImageFont.load_default()
                score_text = f"{score} pts"
                text_width, text_height = draw.textsize(score_text, font=font)
                draw.rectangle([5, img.height - text_height - 10, 20 + text_width, img.height - 8], fill="white")
                draw.text((10, img.height - text_height - 10), score_text, fill="black", font=font)
            canvas.paste(img, (x, y))
        # Add total score in top-right corner of the canvas
        if score_dict:
            total_score = sum(score_dict.values())
            draw = ImageDraw.Draw(canvas)
            try:
                font = ImageFont.truetype("arial.ttf", 36)
            except:
                font = ImageFont.load_default()

            score_text = f"Total: {total_score} pts"
            text_width, text_height = draw.textsize(score_text, font=font)
            margin = 10
            x = canvas.width - text_width - margin
            y = margin

            # Optional: background rectangle for better visibility
            draw.rectangle(
                [x - 5, y - 5, x + text_width + 5, y + text_height + 5],
                fill="white"
            )
            draw.text((x, y), score_text, fill="black", font=font)
        canvas.save(output_path)

    def end_game(self):
        """Handles scoring and display at the end of the game."""
        # print("\n=== 🏁 Game Over! Scoring ===")
        scores = []

        for player in self.players:
            score, card_scores = player.calculate_score()
            # self.display_faraway_cards(
            #     Player.extract_ids(player.played_cards),
            #     Player.extract_ids(player.sanctuaries),
            #     card_path, card_scores, f'faraway_{player.name}.jpg'
            # )
            scores.append((player.name, score))
            # print(f"{player.name}: {score} points")

        # max_score = max(s[1] for s in scores)
        # winners = [name for name, score in scores if score == max_score]

        # if len(winners) == 1:
        #     print(f"\n🏆 Winner: {winners[0]} with {max_score} points!")
        # else:
        #     print(f"\n🤝 It's a tie between: {', '.join(winners)} with {max_score} points each!")

    def load_cards(self):
        """Loads cards from Excel file and separates them by type."""
        df = pd.read_excel(self.card_file, engine='openpyxl')
        normal_cards = []
        sanctuary_cards = []

        for _, row in df.iterrows():
            card = Card(
                id=row['id'],
                points=0 if pd.isna(row.get('points')) else int(row.get('points')),
                color=row.get('color', ''),
                symbols=self.safe_split(row.get('symbols')),
                conditional_effect=self.safe_split(row.get('conditional_effect')),
                activation_requirements=self.safe_split(row.get('activation_requirements')),
                type=row.get('type', '')
            )

            if str(row.get('type', '')).strip().lower() == 'sanctuary':
                sanctuary_cards.append(card)
            else:
                normal_cards.append(card)

        return normal_cards, sanctuary_cards

    def start_game(self):
        """Starts a full game: loads cards, initializes players, and plays 8 rounds."""
        self.deck, self.sanctuary_deck = self.load_cards()
        random.shuffle(self.deck)
        random.shuffle(self.sanctuary_deck)
        self.players = [Player(name) for name in self.player_names]
        self.middle_cards = [self.deck.pop(0) for _ in range(len(self.players) + 1)]

        for _ in range(3):
            for player in self.players:
                if self.deck:
                    player.draw_card(self.deck.pop(0))

        for i in range(8):
            self.play_round(i + 1)

        self.end_game()

    def play_round(self, round_num):
        """Plays a single round including card play, sanctuary logic, and replenishment."""
        for p in self.players:
            p.choose_card_intelligent()

        order = sorted(self.players, key=lambda p: p.current_card.id)

        if round_num > 1:
            for p in order:
                prev = p.played_cards[-2]
                curr = p.current_card
                if curr.id > prev.id:
                    clues = p.count_clues()
                    n = 1 + clues
                    p.sanctuaries_drawn = [self.sanctuary_deck.pop(0) for _ in range(min(n, len(self.sanctuary_deck)))]

        for p in order:
            if round_num < 8:
                p.pick_from_center(self.middle_cards)
            p.choose_sanctuary()
            p.discard_excess_sanctuaries(self.sanctuary_deck)

        if round_num < 8:
            self.middle_cards = [self.deck.pop(0) for _ in range(min(len(self.players) + 1, len(self.deck)))]

    def print_game_state(self):
        """Prints current game state for debugging."""
        print("\n--- Middle Cards ---")
        for card in self.middle_cards:
            print("   ", card)

        print("\n--- Player States ---")
        for player in self.players:
            print(player)
            print("   Played Cards:", player.played_cards)
            print("   Sanctuaries:", player.sanctuaries)
            print("   Hand:", player.hand)


def run_simulations(nb_simulations, player_names, card_file, top_tracker=None):
    all_scores = []

    for i in range(nb_simulations):
        game = Game(player_names, card_file)
        game.start_game()

        for player in game.players:
            total_score, card_scores = player.calculate_score()
            if top_tracker:
                played_ids = Player.extract_ids(player.played_cards)
                sanctuary_ids = Player.extract_ids(player.sanctuaries)
                top_tracker.try_add_game(total_score, played_ids, sanctuary_ids, card_scores, i)

            all_scores.append({player.name: total_score})

    return all_scores
import heapq
class TopGameRecord:
    """Track top scoring games for later image saving."""
    def __init__(self):
        self.top_games = []  # Min-heap of (score, index, data)

    def try_add_game(self, score, played_ids, sanctuary_ids, card_scores, idx):
        entry = (score, idx, played_ids, sanctuary_ids, card_scores)
        if len(self.top_games) < 10:
            heapq.heappush(self.top_games, entry)
        else:
            heapq.heappushpop(self.top_games, entry)

    def save_top_game_images(self, card_path, card_file):
        game = Game(["dummy1", "dummy2"], card_file)
        for rank, (score, idx, played_ids, sanctuary_ids, card_scores) in enumerate(sorted(self.top_games, reverse=True), 1):
            filename = f"top_{rank}.jpg"
            game.display_faraway_cards(played_ids, sanctuary_ids, card_path, card_scores, output_path=filename)
            print(f"✅ Saved image for top score #{rank}: {filename}")

def analyze_results(results):
    """Analyzes simulation results and prints statistics."""
    df = pd.DataFrame(results)
    print("\n=== Résumé des scores sur toutes les simulations ===")
    print(df.describe())

    best_score = df.max().max()
    print(f"\n🔝 Score max atteint sur toutes les parties : {best_score}")

    top_players = df.mean().sort_values(ascending=False)
    print("\n🏅 Moyenne des scores par joueur :")
    print(top_players)


def run_manual_mode(card_file, played_card_ids, sanctuary_card_ids):
    # Load all cards
    df = pd.read_excel(card_file, engine='openpyxl')
    all_cards = {}

    for _, row in df.iterrows():
        card = Card(
            id=row['id'],
            points=0 if pd.isna(row.get('points')) else int(row.get('points')),
            color=row.get('color', ''),
            symbols=[x.strip() for x in str(row.get('symbols')).split(',')] if pd.notna(row.get('symbols')) else [],
            conditional_effect=[x.strip() for x in str(row.get('conditional_effect')).split(',')] if pd.notna(
                row.get('conditional_effect')) else [],
            activation_requirements=[x.strip() for x in
                                     str(row.get('activation_requirements')).split(',')] if pd.notna(
                row.get('activation_requirements')) else [],
            type=row.get('type', '')
        )
        all_cards[card.id] = card

    # Build the manual player
    player = Player("Manual")
    player.played_cards = [all_cards[i] for i in played_card_ids]
    player.sanctuaries = [all_cards[i] for i in sanctuary_card_ids]

    # Score
    total, score_dict = player.calculate_score()
    print(f"\n🧮 Total Score: {total} pts")

    # Output image
    game = Game(["Dummy1", "Dummy2"], card_file)
    game.display_faraway_cards(played_card_ids, sanctuary_card_ids, card_path, score_dict,
                               output_path="manual_output.jpg")
    print("🖼️ Image saved as 'manual_output.jpg'")


# === Run the Game ===
if __name__ == "__main__":
    player_names = ["Alice", "Bob", "Charlie", "Dana"]
    card_file = "../Documentation/faraway_data.xlsx"
    top_games = TopGameRecord()
    # run_manual_mode(card_file, played_card_ids=[5, 12, 7, 13, 28, 43, 38, 23], sanctuary_card_ids=[70, 75, 79, 85])
    results = run_simulations(100, player_names, card_file, top_tracker=top_games)
    analyze_results(results)
    top_games.save_top_game_images(card_path, card_file)
