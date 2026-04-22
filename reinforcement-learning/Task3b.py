#!/usr/bin/env python
# Task 3b: Modified from RL-02-c--CartPole-v0-code3.py
# Change: Replace manual gif-making with RecordVideo + RecordEpisodeStatistics.
#         Runs 100 episodes with TOGGLE policy, records one mp4 per episode.

import gymnasium as gym
import numpy as np
import os

NUM_EPISODES = 100
VIDEO_DIR = "video"

os.makedirs(VIDEO_DIR, exist_ok=True)

base_env = gym.make("CartPole-v0", render_mode="rgb_array")
stats_env = gym.wrappers.RecordEpisodeStatistics(base_env)
env = gym.wrappers.RecordVideo(
    stats_env,
    video_folder=VIDEO_DIR,
    episode_trigger=lambda ep_id: True,  # record EVERY episode
    name_prefix="cartpole-toggle"
)

episode_rewards = []

for episode in range(NUM_EPISODES):
    total_reward = 0.0
    observation, info = env.reset(seed=None)
    action = 0  # start LEFT

    while True:
        # Deterministic toggle policy (same as Task 2)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        action = 1 - action  # toggle each step

        if terminated or truncated:
            ep_info = info.get("episode", {})
            print("Episode {:3d} | Reward: {:.1f} | Length: {} steps".format(
                episode + 1,
                ep_info.get('r', total_reward),
                ep_info.get('l', '?')
            ))
            break

    episode_rewards.append(total_reward)

env.close()

rewards = np.array(episode_rewards)
print("\n" + "=" * 51)
print("  TASK 3b RESULTS -- Toggle Policy + RecordVideo")
print("=" * 51)
print("  Maximum reward : {:.2f}".format(rewards.max()))
print("  Average reward : {:.2f}".format(rewards.mean()))
print("  Std deviation  : {:.2f}".format(rewards.std()))
print("  Videos saved to: {}/".format(VIDEO_DIR))
print("=" * 51)
