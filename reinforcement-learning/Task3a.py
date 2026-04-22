#!/usr/bin/env python
# Task 3a: Modified from RL-02-c--CartPole-v0-code3.py
# Change: Replace the manual gif-making approach with Gymnasium's
#         RecordVideo and RecordEpisodeStatistics wrappers.
#         Runs 100 episodes with RANDOM policy, records one mp4 per episode.

import gymnasium as gym
import numpy as np
import os

NUM_EPISODES = 100
VIDEO_DIR = "video"  # same existing video directory as original code

os.makedirs(VIDEO_DIR, exist_ok=True)

# Instead of manual imageio gif approach from original code,

# Step 1: base env with rgb_array (same render_mode as original)
base_env = gym.make("CartPole-v0", render_mode="rgb_array")

# Step 2: RecordEpisodeStatistics wrapper , tracks reward, length, time per episode
stats_env = gym.wrappers.RecordEpisodeStatistics(base_env)

# Step 3: RecordVideo wrapper , saves one mp4 per episode to video directory
env = gym.wrappers.RecordVideo(
    stats_env,
    video_folder=VIDEO_DIR,
    episode_trigger=lambda ep_id: True,  # record EVERY episode
    name_prefix="cartpole-random"
)

episode_rewards = []

for episode in range(NUM_EPISODES):
    total_reward = 0.0
    observation, info = env.reset(seed=None)

    while True:
        # Random policy (same as original code)
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

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
print("\n" + "=" * 50)
print("  TASK 3a RESULTS -- Random Policy + RecordVideo")
print("=" * 50)
print("  Maximum reward : {:.2f}".format(rewards.max()))
print("  Average reward : {:.2f}".format(rewards.mean()))
print("  Std deviation  : {:.2f}".format(rewards.std()))
print("  Videos saved to: {}/".format(VIDEO_DIR))
print("=" * 50)
