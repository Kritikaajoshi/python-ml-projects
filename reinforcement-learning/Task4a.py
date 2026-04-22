#!/usr/bin/env python
# Task 4a: Different environment with RANDOM policy
# Environment: MountainCar-v0
# A car in a valley must build momentum to reach the flag on the right hill.
# State: [position, velocity]
# Actions: 0=push left, 1=no push, 2=push right
# Reward: -1 per timestep (episode ends at 200 steps or when car reaches goal)

import gymnasium as gym
import numpy as np

NUM_EPISODES = 100

env = gym.make("MountainCar-v0")
episode_rewards = []

for episode in range(NUM_EPISODES):
    total_reward = 0.0
    observation, info = env.reset(seed=None)

    while True:
        # Uniform random policy
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    episode_rewards.append(total_reward)
    print("Episode {:3d} | Total Reward: {:.1f}".format(episode + 1, total_reward))

env.close()

rewards = np.array(episode_rewards)
print("\n" + "=" * 50)
print("  TASK 4a RESULTS -- MountainCar Random Policy")
print("=" * 50)
print("  Maximum reward : {:.2f}".format(rewards.max()))
print("  Average reward : {:.2f}".format(rewards.mean()))
print("  Std deviation  : {:.2f}".format(rewards.std()))
print("=" * 50)
