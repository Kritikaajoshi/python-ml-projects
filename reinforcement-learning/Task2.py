
# Task 2: Modified from RL-02-c--CartPole-v0-code3.py
# Change: Run 100 EPISODES with DETERMINISTIC TOGGLE policy instead of random
# Policy: agent begins pushing LEFT, toggles direction every timestep until episode ends
# Report: max, average, and std deviation of total reward per episode

import gymnasium as gym
import numpy as np

NUM_EPISODES = 100

render_mode = 'rgb_array'
env = gym.make("CartPole-v0", render_mode=render_mode)

episode_rewards = []

for episode in range(NUM_EPISODES):
    total_reward = 0.0
    observation, info = env.reset(seed=None)
    action = 0  

    while True:
        # Deterministic toggle policy 
        # agent begins with pushing the cart left, and each time-step
        # agent toggles the movement until the pole is dropped
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        action = 1 - action  # toggle: 0->1->0->1...

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)
    print("Episode {:3d} | Total Reward: {:.1f}".format(episode + 1, total_reward))

env.close()

rewards = np.array(episode_rewards)
print("\n" + "=" * 48)
print("  TASK 2 RESULTS -- Toggle Policy (100 episodes)")
print("=" * 48)
print("  Maximum reward : {:.2f}".format(rewards.max()))
print("  Average reward : {:.2f}".format(rewards.mean()))
print("  Std deviation  : {:.2f}".format(rewards.std()))
print("=" * 48)
