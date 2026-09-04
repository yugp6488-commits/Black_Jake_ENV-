# Blackjack Reinforcement Learning 🃏🤖

This repository contains Python implementations of reinforcement learning agents trained to play Blackjack using the Gymnasium (`Blackjack-v1`) environment. 

Two distinct Temporal Difference (TD) learning algorithms are included:
1. **Q-Learning** (Off-policy)
2. **SARSA** (On-policy)

## 🌍 Environment Overview

The environment used is `Blackjack-v1` from the [Gymnasium](https://gymnasium.farama.org/) library.

* **Objective:** Beat the dealer's hand without exceeding a total sum of 21.
* **Observation Space (Tuple):** 
  * `player_sum` (int): Player's current hand sum (4 to 21).
  * `dealer_card` (int): Dealer's showing card (1 to 10).
  * `usable_ace` (bool): `True` if the player has an Ace that can be counted as 11 without busting.
* **Action Space (Discrete):**
  * `0`: **Stick/Stand** (Keep the current hand).
  * `1`: **Hit** (Draw another card).
* **Rewards:** 
  * `+1` for winning the hand.
  * `0` for a draw (push).
  * `-1` for losing the hand or busting (exceeding 21).

## 🧠 Algorithms

### 1. Q-Learning
Q-Learning is an **off-policy** algorithm. It updates the Q-values based on the maximum possible reward of the next state, assuming the agent will take the best possible action, regardless of the current exploration policy.

### 2. SARSA (State-Action-Reward-State-Action)
SARSA is an **on-policy** algorithm. It updates the Q-values based on the actual next action chosen by the current epsilon-greedy policy. This makes SARSA slightly more conservative than Q-Learning during training.

## ⚙️ Hyperparameters

The default hyperparameters used for both algorithms:
* `episodes` = 5000 (Number of training games)
* `alpha` = 0.5 (Learning rate)
* `gamma` = 0.99 (Discount factor for future rewards)
* `epsilon` = 0.1 (Exploration rate for the epsilon-greedy policy)

## 🚀 Installation & Usage

1. **Install dependencies:**
   Make sure you have `gymnasium` and `numpy` installed.
   ```bash
   pip install gymnasium numpy
   ```

2. **Run the agents:**
   Execute the python scripts to train the agents. During training, the code occasionally renders the environment (every 1000 episodes) to visualize the progress. After training, a final test run is executed using the fully greedy policy (exploiting the learned Q-table).

## 📊 Results Output
During training, the terminal will output the total reward and episode length every 100 episodes. After training completes, the agent will play a single test game and output the final reward and steps taken.
