import numpy as np
import random
from collections import namedtuple, deque
import os

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
import argparse
from tqdm import tqdm

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make("MountainCar-v0", render_mode="rgb_array")

no_human_action_policy = QNetwork(state_size=2, action_size=3, seed=0).to(device)
no_human_action_policy.load_state_dict(torch.load('./checkpoints/mc_ofrl_v6_s0/dqn.pth', map_location=device))

human_action_policy = QNetwork(state_size=2, action_size=3, seed=0).to(device)
human_action_policy.load_state_dict(torch.load('./checkpoints/mc_ofrl_ha_v6_s0/dqn.pth', map_location=device))

start_x, end_x = -1.2, 0.6
start_y, end_y = -0.07, 0.07

range_x = np.arange(start_x, end_x, 0.1) 
range_y = np.arange(start_y, end_y, 0.01)

for y in range_y:
    arr = np.array([0.3, y], dtype=np.float32)

    # Evaluate the optimal policy
    agent_state_optimal = torch.from_numpy(arr).float().unsqueeze(0).to(device)
    no_human_action_policy.eval()
    with torch.no_grad():
        action_values_noha = no_human_action_policy(agent_state_optimal)
    no_human_action_policy.train()
    action_noha = np.argmax(action_values_noha.cpu().data.numpy())

    # Evaluate the offline policy
    agent_state_offline = torch.from_numpy(arr).float().unsqueeze(0).to(device)
    human_action_policy.eval()
    with torch.no_grad():
        action_values_ha = human_action_policy(agent_state_offline)
    human_action_policy.train()
    action_ha = np.argmax(action_values_ha.cpu().data.numpy())

    # Print comparison
    print(f"Y: {y:.2f},\n No Human Action Policy: {action_values_noha}, Action: {action_noha}, \n Human Action Policy: {action_values_ha}, Action: {action_ha} \n\n")
