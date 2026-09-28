import re
from collections import Counter


class FeatureExtractor:
    def __init__(self, hand_text):
        self.hand_text = hand_text
        self.hero_hand = self._get_hero_hand()
        self.full_board = self._get_board()
        self.position = self.get_position()
        # Pre-calculate the final result once for the whole hand
        self.final_net_result = self._get_net_result()

    def _get_hero_hand(self):
        match = re.search(r"Dealt to Hero \[(.*?)\]", self.hand_text)
        return match.group(1).split() if match else []

    def _get_board(self):
        match = re.search(r"Board \[(.*?)\]", self.hand_text)
        return match.group(1).split() if match else []

    def get_position(self):
        try:
            btn = re.search(r"Seat #(\d+) is the button", self.hand_text).group(1)
            hero = re.search(r"Seat (\d+): Hero", self.hand_text).group(1)
            return (int(hero) - int(btn)) % 6
        except:
            return 0

    def check_potential(self, cards):
        if not cards: return {"max_suit": 0, "max_consec": 0, "has_gutshot": 0}
        suits = [c[-1] for c in cards]
        max_suit = max(Counter(suits).values()) if suits else 0

        ranks = sorted(list(set(["..23456789TJQKA".find(c[0]) for c in cards])))
        if 14 in ranks: ranks.insert(0, 1)

        max_c = 1;
        curr_c = 1;
        has_gut = 0
        for i in range(len(ranks) - 1):
            if ranks[i + 1] - ranks[i] == 1:
                curr_c += 1
            else:
                max_c = max(max_c, curr_c); curr_c = 1
        max_c = max(max_c, curr_c)

        for i in range(len(ranks) - 3):
            if ranks[i + 3] - ranks[i] == 4: has_gut = 1
        return [max_suit/5, max_c/5, has_gut]
        #return {"max_suit": max_suit, "max_consec": max_c, "has_gutshot": has_gut}

    def _get_net_result(self):
        winner = re.search(r"Hero .*?won \(\$([\d.]+)\)", self.hand_text)
        # Tracks every cent Hero put in (blinds, bets, calls, raises)
        spent = sum(map(float, re.findall(r"Hero: (?:calls|raises|bets|posts).*?\$([\d.]+)", self.hand_text)))
        if winner:
            return float(winner.group(1))/2
        else:
            return -spent

    def extract_data(self):
        sections = re.split(r"\*\*\* (HOLE CARDS|FLOP|TURN|RIVER) \*\*\*", self.hand_text)[1:]
        results = []
        current_pot = sum(map(float, re.findall(r"posts .*? \$([\d.]+)", self.hand_text)))
        # Standard Poker Rank Mapping
        RANK_MAP = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        Stage_map = {'HOLE CARDS': 0, 'FLOP': 1, 'TURN': 2, 'RIVER': 3}

        for i in range(1, len(sections), 2):
            stage_name = sections[i - 1]
            content = sections[i]
            board_state = self.full_board[:3] if stage_name == "FLOP" else self.full_board[
                :4] if stage_name == "TURN" else self.full_board[:5] if stage_name == "RIVER" else []

            # 1. NEW REGEX: Capture the whole line to check for "to $"
            all_actions = re.finditer(r"(\w+): (?!posts)(\w+)(.*)", content)

            street_max_bet = 0.02 if stage_name == "HOLE CARDS" else 0.0
            hero_committed_this_street = 0.0

            for match in all_actions:
                player = match.group(1)
                action_type = match.group(2).lower()
                line_tail = match.group(3)
                current = Stage_map[stage_name]
                one_hot_stage = [0] * 4
                one_hot_stage[current] = 1

                if action_type in ["shows", "collected", "chooses", "pays"]: continue

                # 2. LOGIC: Extract the ACTUAL total bet level
                # If "to $0.08" exists, use 0.08. Otherwise, look for "$0.02".
                to_match = re.search(r"to \$([\d.]+)", line_tail)
                amt_match = re.search(r"\$([\d.]+)", line_tail)

                if to_match:
                    act_total = float(to_match.group(1))
                elif amt_match:
                    act_total = float(amt_match.group(1))
                else:
                    act_total = street_max_bet if action_type == "calls" else 0.0

                if player == "Hero":
                    # Facing Ratio: Price to stay in vs Pot before Hero acts
                    facing_ratio = (street_max_bet - hero_committed_this_street) / current_pot if current_pot > 0 else 0

                    # Hero Bet Ratio: Total NEW money Hero puts in vs Pot before Hero acts
                    # In a Raise to $0.08, if Hero already put in $0.02, he adds $0.06
                    new_money = act_total - hero_committed_this_street
                    hero_bet_ratio = new_money / current_pot if current_pot > 0 else 0

                    results.append([
                         [round(RANK_MAP[k[0]]/14, 3) for k in self.hero_hand],
                         one_hot_stage,
                         ([round(RANK_MAP[k[0]]/14, 3) for k in board_state] + [0] * 5)[:5],
                         round(self.position/6, 3),
                         self.check_potential(self.hero_hand + board_state),
                         round(facing_ratio, 3),
                         1 if action_type == "folds" else 0,
                         round(hero_bet_ratio, 3),
                         self.final_net_result
                    ])

                    hero_committed_this_street = act_total
                    if action_type == "folds": return results

                # 3. Update Pot and Street Max for the next player
                # Add only the NEW money this player put into the pot
                # (Current total commitment - what they previously contributed)
                # For simplicity here: we add the increment
                if action_type != "folds":
                    increment = act_total if action_type in ["bets", "raises", "calls"] else 0
                    # Note: A real tracker would subtract the player's previous bet,
                    # but for 0.01/0.02, act_total is usually the increment except in 'raises to'
                    current_pot += (act_total - (street_max_bet if action_type == "raises" else 0))

                    if act_total > street_max_bet:
                        street_max_bet = act_total

        return results



# Execution
if __name__ == "__main__":
    extractor = FeatureExtractor("""Poker Hand #HD2531268066: Hold'em No Limit ($0.01/$0.02) - 2025/10/12 13:23:33
    Table 'NLHWhite92' 6-max Seat #6 is the button
    Seat 1: 1a354f1e ($2.06 in chips)
    Seat 2: 823ba7ee ($0.61 in chips)
    Seat 3: Hero ($1.73 in chips)
    Seat 4: 467b408e ($2.75 in chips)
    Seat 5: e0d5be76 ($1.41 in chips)
    Seat 6: f34c7df ($1.97 in chips)
    1a354f1e: posts small blind $0.01
    823ba7ee: posts big blind $0.02
    *** HOLE CARDS ***
    Dealt to 1a354f1e 
    Dealt to 823ba7ee 
    Dealt to Hero [9c Qs]
    Dealt to 467b408e 
    Dealt to e0d5be76 
    Dealt to f34c7df 
    Hero: raises $0.02 to $0.04
    467b408e: folds
    e0d5be76: folds
    f34c7df: folds
    1a354f1e: folds
    823ba7ee: calls $0.02
    *** FLOP *** [Kh Qc 5h]
    823ba7ee: checks
    Hero: bets $0.03
    823ba7ee: calls $0.03
    *** TURN *** [Kh Qc 5h] [3s]
    823ba7ee: checks
    Hero: bets $0.08
    823ba7ee: calls $0.08
    *** RIVER *** [Kh Qc 5h 3s] [Ac]
    823ba7ee: checks
    Hero: checks
    823ba7ee: shows [7d Qh] (a pair of Queens)
    Hero: shows [9c Qs] (a pair of Queens)
    *** SHOWDOWN ***
    Hero collected $0.3 from pot
    *** SUMMARY ***
    Total pot $0.31 | Rake $0.01 | Jackpot $0 | Bingo $0 | Fortune $0 | Tax $0
    Board [Kh Qc 5h 3s Ac]
    Seat 1: 1a354f1e (small blind) folded before Flop
    Seat 2: 823ba7ee (big blind) showed [7d Qh] and lost with a pair of Queens
    Seat 3: Hero showed [9c Qs] and won ($0.3) with a pair of Queens
    Seat 4: 467b408e folded before Flop (didn't bet)
    Seat 5: e0d5be76 folded before Flop (didn't bet)
    Seat 6: f34c7df (button) folded before Flop (didn't bet)""")
    data = extractor.extract_data()
    for k in data:
        print(k)
