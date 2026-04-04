import gymnasium as gym
import pickle
from collections import defaultdict
import time
from lib import Statistics

ACTION_MAP = {0: "STICK", 1: "HIT"}

def load_model(filename="blackjack_q.pkl"):
    with open(filename, "rb") as f:
        data, epsilon, stats = pickle.load(f)
    Q = defaultdict(lambda: [0.0, 0.0], data)
    return Q

def run_game(Q, money, bet, exponential_betting, delay):
    env = gym.make("Blackjack-v1")
    state, _ = env.reset()
    done = False

    while not done:
        player_sum, dealer_card, usable_ace = state
        action = Q[state].index(max(Q[state]))
        action_str = ACTION_MAP[action]
        print(f"Player total: {player_sum} | Dealer showing: {dealer_card} | Action: {action_str}")

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        time.sleep(delay)

    env.close()
    return reward

# ----- Generalized simulator -----
def blackjack_simulator(n_games=None, starting_money=100, base_bet=1, exponential_betting=False, delay=0.3):
    Q = load_model()
    money = starting_money if n_games is None else n_games * base_bet
    bet = base_bet
    max_money = money
    max_bet = bet
    stats = Statistics()

    game_number = 0

    # Loop until bankrupt (if n_games=None) or fixed number of games
    while money > 0 and (n_games is None or game_number < n_games):
        game_number += 1
        print(f"\n=== Game {game_number} | Money: {money:.1f} | Bet: {bet:.1f} ===")
        reward = run_game(Q, money, bet, exponential_betting, delay)

        money += reward * bet
        stats.games += 1
        if reward > 0:
            stats.wins += 1
            result = "WIN"
            if exponential_betting:
                bet = base_bet
        elif reward == 0:
            stats.draws += 1
            result = "DRAW"
        else:
            stats.losses += 1
            result = "LOSS"
            if exponential_betting:
                bet *= 2

        

        # All or nothing
        if bet > money:
            print("All in!")
            bet = money

        max_money = max(max_money, money)
        max_bet = max(max_bet, bet)

        print(f"End of game: {result} | Money: {money:.1f} | Next bet: {bet:.1f}")

    print("\n--- Simulation ended ---")
    print(f"Games played: {stats.games}")
    print(f"Bankrupt" if money <= 0 else f"Remaining money: {money:.1f}")
    print(f"Highest money held: {max_money:.1f}")
    print(f"Highest bet placed: {max_bet:.1f}")
    print(f"Wins: {stats.wins}, Draws: {stats.draws}, Losses: {stats.losses}")
    win_rate = stats.wins / stats.games if stats.games > 0 else 0
    print(f"Win rate: {win_rate:.3f}")

# Wrappers
def play_n_games(n_games=20, delay=0.8, base_bet=1, exponential_betting=False):
    blackjack_simulator(n_games=n_games, starting_money=None, base_bet=base_bet,
                        exponential_betting=exponential_betting, delay=delay)

def simulate_bankroll(starting_money=100, base_bet=1, delay=0.3, exponential_betting=False):
    blackjack_simulator(n_games=None, starting_money=starting_money, base_bet=base_bet,
                        exponential_betting=exponential_betting, delay=delay)

# ----- Main -----
if __name__ == "__main__":
    # play_n_games(n_games=20, delay=0.5, base_bet=1, exponential_betting=False)


    simulate_bankroll(starting_money=100, base_bet=1, delay=0.2, exponential_betting=True)