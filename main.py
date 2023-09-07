import os

import highway_env.envs.common.action
import numpy as np
import gymnasium as gym
import random
import math
import torch
import matplotlib.pyplot as plt
from highway_env import envs
from agents.dqn_agent import DQNAgent
from config import case1



def seed_env(random_seed):
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


seed_env(42)

EPISODES = 100
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
GAMMA = 0.8
LR = 0.001
BATCH_SIZE = 32
STATE_DIM = 25
ACTION_DIM = 5

env = envs.HighwayEnv(config=case1)  # gym.make("highway-v0")
env.action_space.seed(42)

agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, lr=LR,
                 eps_end=EPS_END, eps_start=EPS_START, eps_decay=EPS_DECAY,
                 batch_size=BATCH_SIZE, gamma=GAMMA)
score_history = []

for e in range(1, EPISODES + 1):
    state, _ = env.reset(seed=42)
    steps = 0
    rewards = 0
    while True:
        # env.render()
        state = torch.FloatTensor(state).reshape(-1, 5, 5)
        action = agent.action(state)
        next_state, reward, done, _, info = env.step(action.item())
        if info['speed'] < 11:
            done = True
        if done:
            print(f"Speed: {info['speed']}, Reward : {reward}")
            reward = -1
        agent.memorize(state, action, reward, next_state)
        agent.learn()
        state = next_state
        rewards += reward
        steps += 1
        if done:
            print("Eposide:{0} Score: {1} Reward {2}".format(e, steps, rewards))
            print("-" * 100)
            score_history.append(rewards)
            break

plt.plot(score_history)
plt.ylabel("score")
plt.show()

state1, _ = env.reset(seed=42)
with torch.no_grad():
    while True:
        # env.render()
        state1 = torch.FloatTensor(state1).reshape(-1, 5, 5)
        action = agent.action(state1)
        state1, reward, done, _, _ = env.step(action.item())
