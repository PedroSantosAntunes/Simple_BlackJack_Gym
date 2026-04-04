import gymnasium as gym
import pickle
from collections import defaultdict
import time
from lib import Statistics  # assuming your Statistics class is here

# ---- Load Q-table ----
def load_model(filename="blackjack_q.pkl"):
    with open(filename, "rb") as f:
        data, epsilon, stats = pickle.load(f)
    Q = defaultdict(lambda: [0.0, 0.0], data)
    return Q

# Map action int → string
ACTION_MAP = {0: "STICK", 1: "HIT"}

def simulate_bankroll(starting_money=100, base_bet=1, delay=0.3, exponential_betting=False):
    env = gym.make("Blackjack-v1")
    Q = load_model()

    money = starting_money
    bet = base_bet
    max_money = money
    max_bet = bet

    stats = Statistics()

    while money > 0:
        state, _ = env.reset()
        done = False
        stats.games += 1

        print(f"\n=== Game {stats.games} | Current money: {money:.1f} | Current bet: {bet:.1f} ===")

        while not done:
            player_sum, dealer_card, usable_ace = state

            # greedy policy
            action = Q[state].index(max(Q[state]))
            action_str = ACTION_MAP[action]

            print(f"Player total: {player_sum} | Dealer showing: {dealer_card} | Action: {action_str}")

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            time.sleep(delay)

        # update stats
        if reward > 0:
            stats.wins += 1
            if exponential_betting:
                bet = base_bet
            result = "WIN"
        elif reward == 0:
            stats.draws += 1
            result = "DRAW"
            # bet unchanged
        else:
            stats.losses += 1
            result = "LOSS"
            if exponential_betting:
                bet *= 2

        # update money
        money += reward * bet

        # ensure bet doesn't exceed current money
        if bet > money:
            bet = money

        # track maximums
        max_money = max(max_money, money)
        max_bet = max(max_bet, bet)

        print(f"End of game: {result} | money={money:.1f} | next bet={bet:.1f}")

    env.close()

    # final summary
    print("\n--- Simulation ended ---")
    print(f"Bankrupt after {stats.games} games")
    print(f"Highest money held: {max_money:.1f}")
    print(f"Highest bet placed: {max_bet:.1f}")
    print(f"Wins: {stats.wins}, Draws: {stats.draws}, Losses: {stats.losses}")
    win_rate = stats.wins / stats.games if stats.games > 0 else 0
    print(f"Win rate: {win_rate:.3f}")

if __name__ == "__main__":
    simulate_bankroll(starting_money=100, base_bet=1, delay=0.3, exponential_betting=True)