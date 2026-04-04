import gymnasium as gym
import pickle
from collections import defaultdict
import time
from lib import Statistics

def load_model(filename="blackjack_q.pkl"):
    with open(filename, "rb") as f:
        data, epsilon, stats = pickle.load(f)
    Q = defaultdict(lambda: [0.0, 0.0], data)
    return Q

ACTION_MAP = {0: "STICK", 1: "HIT"}

def play(n_games=20, delay=0.8, base_bet=1, exponential_betting=False):
    env = gym.make("Blackjack-v1")
    Q = load_model()

    wins = 0
    games = 0
    money = n_games * base_bet
    bet = base_bet

    for i in range(1, n_games + 1):
        if money <= 0:
            print("Whomp whomp")
            break

        if money < bet:
            print("All or nothing!")
            bet = money


        state, _ = env.reset()
        done = False
        print(f"\n=== Game {i} | Current money: {money:.1f} | Current bet: {bet:.1f} ===")

        while not done:
            player_sum, dealer_card, usable_ace = state

            action = Q[state].index(max(Q[state]))
            action_str = ACTION_MAP[action]

            print(f"Player total: {player_sum} | Dealer showing: {dealer_card} | Action: {action_str}")

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            time.sleep(delay)

        # update stats and money
        games += 1
        money += reward * bet

        if reward > 0:
            wins += 1
            result = "WIN"
            if exponential_betting:
                bet = base_bet
        elif reward == 0:
            result = "DRAW"
        else:
            result = "LOSS"
            if exponential_betting:
                bet *= 2 # try to recover losses

        
        win_rate = wins / games
        print(f"Game result: {result} | Running win rate: {win_rate:.3f} | Money: {money:.1f}")

    env.close()

if __name__ == "__main__":
    play(n_games=20, delay=0.8, base_bet=1, exponential_betting=True)