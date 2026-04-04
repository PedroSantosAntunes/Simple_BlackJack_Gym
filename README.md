# Blackjack Q-Learning Simulator

This project implements a **Q-Learning agent** for the `Blackjack-v1` environment in [Gymnasium](https://gymnasium.farama.org/environments/toy_text/blackjack/). The agent learns an optimal policy for the simplified Blackjack game, and the project provides interactive modes for playing and simulating bankroll management.  

## Environment

The environment is [Blackjack-v1](https://gymnasium.farama.org/environments/toy_text/blackjack/), a simplified, turn-based version of Blackjack designed for reinforcement learning research. Key aspects include:

- **State representation**: Each state is a tuple `(player_sum, dealer_card, usable_ace)`:
  - `player_sum`: the total value of the player’s hand (11–21+).  
  - `dealer_card`: the dealer’s visible card (1–10).  
  - `usable_ace`: a boolean indicating if the player has an ace counted as 11 without busting.  
- **Action space**: Two discrete actions:
  - `STICK (0)`: the player stops taking cards and ends their turn.  
  - `HIT (1)`: the player draws another card.  
- **Rewards**: At the end of a round:
  - `+1` for a win  
  - `0` for a draw  
  - `-1` for a loss  

### Nuances of the environment

- The environment is **partially stochastic**: card draws are random, but the state fully summarizes the player’s hand and dealer’s visible card.  
- **Finite, discrete state space**: The combination of `player_sum`, `dealer_card`, and `usable_ace` is relatively small, which makes Q-Learning feasible.  
- **Simplified dealer rules**: The dealer hits until reaching a total of 17 or higher, which is deterministic.  
- **Blackjack and bust handling**: If the player or dealer exceeds 21, they bust and the game ends immediately.  
- **Endless deck**: The deck is effectively infinite, so cards are drawn with replacement. This makes **card counting impossible**, and the probability of each card is constant throughout the game.  
- **Optimal policy feasibility**: Because the state space is small and all relevant information for decision-making is encoded in the state tuple, Q-Learning can reliably converge to the optimal policy. The main source of randomness is card drawing, but the agent can still estimate expected returns for each state-action pair accurately over many episodes.

This environment abstracts away complex Blackjack rules (like splitting or doubling) while keeping the core strategic challenge, making it an ideal testbed for reinforcement learning algorithms.

## Q-Learning Agent

The agent uses a **Q-table** mapping `(player_sum, dealer_card, usable_ace)` → `[value_stick, value_hit]`.  

- **Learning rate (`alpha`)**: controls how quickly the Q-values are updated.  
- **Discount factor (`gamma`)**: accounts for the expected future reward.  
- **Epsilon-greedy exploration (`epsilon`)**: allows the agent to explore initially and gradually shift to exploiting the learned policy.  

Due to the simplicity of the environment and the small number of possible states, Q-Learning can **reach the optimal policy** after sufficient training episodes.  

## Features

### 1. Play a fixed number of games

- User can specify the number of games (`n_games`) and the base bet.  
- The agent uses its learned Q-table to make decisions.  
- Optionally, **martingale-style exponential betting** can be enabled, where the bet doubles after each loss and resets after a win.  
- Tracks:  
  - Player total card value  
  - Dealer visible card value  
  - Action chosen  
  - Money and running win rate  

### 2. Bankroll simulation until bankruptcy

- User can specify starting money and base bet.  
- Games continue until the player goes bankrupt.  
- Tracks:  
  - Highest money held  
  - Highest bet placed  
  - Total games survived  
  - Wins, draws, and losses  
- Optionally, exponential betting can be used to simulate risk recovery strategies.  

## Saving and Loading the Q-Table

The project uses **pickle** to save the Q-table, exploration value, and statistics. This allows:

- Pausing and resuming training.  
- Using a pre-trained model for playing or simulation without retraining.  

## How to Use

1. **Train the Q-Learning agent** using the training script.  
2. **Play fixed games**:
```python
play_n_games(n_games=20, base_bet=1, exponential_betting=True)
```
2. **Simulate bankroll until bankruptcy**:
```python
simulate_bankroll(starting_money=100, base_bet=1, exponential_betting=True)
```

## References

- [Gymnasium Blackjack Environment](https://gymnasium.farama.org/environments/toy_text/blackjack/)  
- Sutton, R.S., Barto, A.G. *Reinforcement Learning: An Introduction*, 2nd Edition