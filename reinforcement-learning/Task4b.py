#!/usr/bin/env python
# Task 4b: Different environment with DETERMINISTIC policy
# Environment: MountainCar-v0 (same as Task 4a)
#
# Deterministic policy chosen: Energy-Pump (velocity matching)
# Intuition: push in the same direction the car is already moving.
# This builds momentum each swing, like pumping a playground swing.
#   - If velocity > 0 (moving right) -> push right (action = 2)
#   - If velocity <= 0 (moving left or stopped) -> push left (action = 0)
# This is more rewarding than random because the car actually reaches the goal.

import gymnasium as gym
import numpy as np

NUM_EPISODES = 100

env = gym.make("MountainCar-v0")
episode_rewards = []

for episode in range(NUM_EPISODES):
    total_reward = 0.0
    observation, info = env.reset(seed=None)

    while True:
        position, velocity = observation  # unpack state

        # Deterministic energy-pump policy (replaces random action)
        if velocity > 0:
            action = 2  # moving right -> push right
        else:
            action = 0  # moving left or stopped -> push left

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)
    print("Episode {:3d} | Total Reward: {:.1f}".format(episode + 1, total_reward))

env.close()

rewards = np.array(episode_rewards)
print("\n" + "=" * 53)
print("  TASK 4b RESULTS -- MountainCar Energy-Pump Policy")
print("=" * 53)
print("  Maximum reward : {:.2f}".format(rewards.max()))
print("  Average reward : {:.2f}".format(rewards.mean()))
print("  Std deviation  : {:.2f}".format(rewards.std()))
print("=" * 53)
