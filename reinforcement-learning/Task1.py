#!/usr/bin/env python
# Task 1: Modified from RL-02-c--CartPole-v0-code3.py
# Change: Run 100 EPISODES (not 500 timesteps) with random policy
# Report: max, average, and std deviation of total reward per episode

import gymnasium as gym
import numpy as np

# Number of episodes
NUM_EPISODES = 100

render_mode = 'rgb_array'
env = gym.make("CartPole-v0", render_mode=render_mode)

episode_rewards = []  # store total reward per episode

for episode in range(NUM_EPISODES):
    total_reward = 0.0
    observation, info = env.reset(seed=None)

    while True:
        # Random policy: same as original code
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)
    print("Episode {:3d} | Total Reward: {:.1f}".format(episode + 1, total_reward))

env.close()

# Report statistics
rewards = np.array(episode_rewards)
print("\n" + "=" * 45)
print("  TASK 1 RESULTS -- Random Policy (100 episodes)")
print("=" * 45)
print("  Maximum reward : {:.2f}".format(rewards.max()))
print("  Average reward : {:.2f}".format(rewards.mean()))
print("  Std deviation  : {:.2f}".format(rewards.std()))
print("=" * 45)
