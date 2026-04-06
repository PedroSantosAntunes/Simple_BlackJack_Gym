import gymnasium as gym
import random
from collections import defaultdict
import pickle
from lib import Statistics

def save_model(Q, epsilon,stats, filename="blackjack_q.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((dict(Q), epsilon, stats), f)

def load_model(filename="blackjack_q.pkl"):
    try:
        with open(filename, "rb") as f:
            data, epsilon, stats = pickle.load(f)
        return defaultdict(lambda: [0.0, 0.0], data), epsilon, stats
    except:
        return defaultdict(lambda: [0.0, 0.0]), 1.0, Statistics()

def train():
    env = gym.make("Blackjack-v1")

    Q, epsilon, stats = load_model()

    n_episodes = 500_000
    alpha = 0.1
    gamma = 0.99
    epsilon_decay = 0.99995
    epsilon_min = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = Q[state].index(max(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward if done else reward + gamma * max(Q[next_state])
            Q[state][action] += alpha * (target - Q[state][action])

            state = next_state

        # stats update (end of episode)
        stats.games += 1
        if reward == 1:
            stats.wins += 1
            result = "WIN"
        elif reward == 0:
            stats.draws += 1
            result = "DRAW"
        else:
            stats.losses += 1
            result = "LOSS"

        win_rate = stats.wins / stats.games

        print(f"Game {stats.games} | {result} | Win rate: {win_rate:.3f} | Epsilon: {epsilon:.3f}")

        # decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if episode % 10_000 == 0 and episode > 0:
            save_model(Q, epsilon, stats)

    return Q


if __name__ == "__main__":
    train()