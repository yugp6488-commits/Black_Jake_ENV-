import gymnasium as gym
import numpy as np
import random

gamma = 0.99
epsilon = 0.1
alpha = 0.5
episodes = 5000

def epsilon_greedy(state, q_table, env):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        player_sum, dealer_card, usable_ace = state
        return np.argmax(q_table[player_sum, dealer_card, usable_ace])

Q = np.zeros((32, 11, 2, 2))

for episode in range(episodes):
    render = (episode % 1000 == 0)

    if render:
        env = gym.make("Blackjack-v1", render_mode="human")
    else:
        env = gym.make("Blackjack-v1")

    done = False
    state, _ = env.reset()

    episode_len = 0
    tot_reward = 0

    while not done:
        action = epsilon_greedy(state, Q, env)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        player_sum, dealer_card, usable_ace = state
        next_player_sum, next_dealer_card, next_usable_ace = next_state

       
        Q[player_sum, dealer_card, usable_ace, action] += alpha * (
            reward + gamma * np.max(Q[next_player_sum, next_dealer_card, next_usable_ace]) 
            - Q[player_sum, dealer_card, usable_ace, action]
        )
        
        state = next_state
        episode_len += 1
        tot_reward += reward

    if (episode + 1) % 100 == 0:
        print(f"episode = {episode+1}/{episodes} & total reward = {tot_reward} & episode len = {episode_len}")

    env.close()

test_env = gym.make("Blackjack-v1")
state, _ = test_env.reset()
done = False
total_test_reward = 0
test_episode_len = 0

print("Starting Final Test Run...")

while not done:
    player_sum, dealer_card, usable_ace = state
    action = np.argmax(Q[player_sum, dealer_card, usable_ace])
    next_state, reward, terminated, truncated, _ = test_env.post_action_if_needed if hasattr(test_env, 'post_action_if_needed') else test_env.step(action) 
    next_state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    state = next_state
    total_test_reward += reward
    test_episode_len += 1

print(f"Test Run Completed! Total Reward = {total_test_reward}, Steps Taken = {test_episode_len}")
test_env.close()