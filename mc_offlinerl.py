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
from PIL import Image, ImageDraw, ImageFont

def detect_catastrophe(state, action, force=0.001, gravity=0.0025): # accept state here
    tuple_state = tuple(state)
    position = tuple_state[0]
    velocity = tuple_state[1]

    new_velocity = np.clip(velocity + (action - 1) * force - np.cos(3 * position) * gravity, -0.07, 0.07)
    new_position = np.clip(position + new_velocity, -1.2, 0.6)
    
    return new_position < -1.15

def is_catastrophe(state): # accept state here
    tuple_state = tuple(state)
    position = tuple_state[0]
    return position < -1.15

# Function to discretize the continuous state space
# Note: this function is not used
def discretize_state(state):
    discretized_state = (state - env.observation_space.low) * np.array([20, 20])
    return tuple(discretized_state.astype(int))

def frame_write(frame):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((100, 100), "catastrophe", fill=(255, 0, 0)) #, font=font)
    frame_with_text = np.array(img)
    return frame_with_text

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

# Note: this replay buffer does not have a maximum buffer_size
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque()
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

# Note: this class is not used
class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        if run != None:
            wandb.log({"loss": loss})

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# Setup
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="set the name of the experiment")
parser.add_argument("--numeps", type=int, help="sets number of episodes")
parser.add_argument("--numits", type=int, help="sets number of iterations")
parser.add_argument("--seed", type=int, help="sets seed")
parser.add_argument("--penalty", type=int, help="sets penalty")
parser.add_argument("--addha", help="sets human actions to be true", action="store_true")
parser.add_argument("--wandb", help="sets wandb to be true", action="store_true")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = gym.make("MountainCar-v0", render_mode="rgb_array")

viewport_width = 600  # VIEWPORT_W
viewport_height = 400  # VIEWPORT_H
scale = 30.0  # SCALE

BUFFER_SIZE = int(1e5)          # replay buffer size
BATCH_SIZE = 64                 # minibatch size
GAMMA = 0.99                    # discount factor
TAU = 1e-3                      # for soft update of target parameters
LR = 5e-4                       # learning rate
UPDATE_EVERY = 4                # how often to update the network

EXPERIMENT_NAME = args.name     # set experiment name
NUMEPS = args.numeps            # number of episodes
NUMITS = args.numits            # number of iterations
SEED = args.seed                # seed
PENALTY = args.penalty          # penalty (positive in argument, set to be negative below)
ADD_HA = args.addha             # set if adding human actions
WANDB = args.wandb              # log using wandb

wandb_hyperparameters = {
    'env_name': 'MountainCar-v0',
    'experiment_name': EXPERIMENT_NAME,
    'n_episodes': NUMEPS,
    'n_iterations': NUMITS,
    'batch_size': BATCH_SIZE,
    'gamma': GAMMA,
    'tau': TAU,
    'learning_rate': LR,
    'update_every': UPDATE_EVERY,
    'penalty': PENALTY,
    'seed': SEED,
    'add_ha': ADD_HA,
}

run = wandb.init(project="modified_hirl", name=EXPERIMENT_NAME, config=wandb_hyperparameters) if WANDB else None

# Optimal Policy
optimal_policy = QNetwork(state_size=2, action_size=3, seed=SEED).to(device)
optimal_policy.load_state_dict(torch.load('/home/jaiv/offlinerl_dqn/checkpoints/mc_hirl_v1_s0/mc_dqn_2500.pth', map_location=device))
optimal_policy.eval()

# Note: this replay buffer does not have a maximum buffer_size
optimal_policy_buffer = ReplayBuffer(action_size=3, buffer_size=-1, batch_size=64, seed=SEED)

num_episodes_to_run = NUMEPS
max_t = 200
total_test_reward = 0
num_test_catastrophes = 0
penalty = float(-PENALTY)
flag = 0

if not ADD_HA: # WITHOUT human actions in the buffer
    for episode in tqdm(range(num_episodes_to_run), desc="Collecting experiences"):
        flag = 0
        state, _ = env.reset()
        for i in range(max_t):
            # from agent act:
            agent_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = optimal_policy(agent_state)
            action = np.argmax(action_values.cpu().data.numpy())
            robot_action = action

            if detect_catastrophe(state, action):
                flag = 1
                action = 2

            if is_catastrophe(state):
                num_test_catastrophes += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if flag == 1:
                optimal_policy_buffer.add(state, robot_action, penalty, next_state, done)
                reward = penalty
                flag = 0
            else:
                optimal_policy_buffer.add(state, robot_action, reward, next_state, done)

            state = next_state
            total_test_reward += reward

            if done:
                break
else: # WITH human actions in the buffer
    for episode in tqdm(range(num_episodes_to_run), desc="Collecting experiences"):
        flag = 0
        state, _ = env.reset()
        for i in range(max_t):
            # from agent act:
            agent_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = optimal_policy(agent_state)
            action = np.argmax(action_values.cpu().data.numpy())
            robot_action = action

            if detect_catastrophe(state, action):
                flag = 1
                action = 2

            if is_catastrophe(state):
                num_test_catastrophes += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if flag == 1:
                optimal_policy_buffer.add(state, robot_action, penalty, next_state, done)
                optimal_policy_buffer.add(state, action, reward, next_state, done)
                reward = penalty
                flag = 0
            else:
                optimal_policy_buffer.add(state, robot_action, reward, next_state, done)

            state = next_state
            total_test_reward += reward

            if done:
                break

average_test_catastrophes = num_test_catastrophes / num_episodes_to_run
average_test_reward = total_test_reward / num_episodes_to_run

print(f"Average Test Catastrophes: {average_test_catastrophes}")
print(f"Average Test Reward: {average_test_reward}")

print(f"Size of Buffer: {optimal_policy_buffer.__len__()}")

# # Offline Agent
gamma = GAMMA
offline_policy = QNetwork(state_size=2, action_size=3, seed=SEED).to(device)
# target_policy = QNetwork(state_size=2, action_size=3, seed=SEED).to(device)
# target_policy.load_state_dict(offline_policy.state_dict())
offline_policy.train()

optimizer = optim.Adam(offline_policy.parameters(), lr=LR)

for i_iteration in range(NUMITS):
    states, actions, rewards, next_states, dones = optimal_policy_buffer.sample()

    # Get max predicted Q values (for next states) from target model
    # Q_targets_next = offline_policy(next_states).detach().max(1)[0].unsqueeze(1)
    Q_targets_next = offline_policy(next_states).max(1)[0].unsqueeze(1)
    # Compute Q targets for current states
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

    # Get expected Q values from local model
    Q_expected = offline_policy(states).gather(1, actions)

    # Compute loss
    loss = F.mse_loss(Q_expected, Q_targets)
    # Minimize the loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if run != None:
        wandb.log({"loss": loss})

    # if i_iteration % 5 == 0:
    #     target_policy.load_state_dict(offline_policy.state_dict())

    # for test_score, test_catastrophe, and render logging:
    if i_iteration % (NUMITS * 0.01) == 0 and run != None:
        frames = []
        num_test_episodes = 10
        total_test_reward = 0
        num_test_catastrophes = 0
        flag = 0

        for episode in range(num_test_episodes):
            state, _ = env.reset()
            for i in range(max_t):
                agent_state = torch.from_numpy(state).float().unsqueeze(0).to(device)
                offline_policy.eval()
                with torch.no_grad():
                    action_values = offline_policy(agent_state)
                offline_policy.train()

                action = np.argmax(action_values.cpu().data.numpy())

                if is_catastrophe(state):
                    num_test_catastrophes += 1
                    flag = 1
                    frame = frame_write(env.render())
                else:
                    frame = env.render()

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if flag == 1:
                    reward = penalty
                    flag = 0

                frames.append(frame)

                state = next_state
                total_test_reward += reward

                if done:
                    break

        if i_iteration % (NUMITS * 0.04) == 0:
            video_frames = np.array(frames)
            video_frames = np.transpose(video_frames, (0, 3, 1, 2))
            wandb.log({"Render": wandb.Video(video_frames, fps=15, format="mp4")})

        average_test_catastrophes = num_test_catastrophes / num_test_episodes
        average_test_reward = total_test_reward / num_test_episodes
        wandb.log({"test_catastophe": average_test_catastrophes, "test_score": average_test_reward})

# ignore old code
############

# offline_agent = Agent(state_size=8, action_size=4, seed=SEED)

# for i_iteration in range(NUMITS):
#     states, actions, rewards, next_states, dones = optimal_policy_buffer.sample()

#     # Assuming the tensors are on CUDA, move them to CPU for processing
#     states = states.cpu()
#     actions = actions.cpu()
#     rewards = rewards.cpu()
#     next_states = next_states.cpu()
#     dones = dones.cpu()

#     # Convert tensors to NumPy arrays if necessary. This depends on how offline_agent.step() expects the data
#     states_np = states.numpy()
#     actions_np = actions.numpy()
#     rewards_np = rewards.numpy()
#     next_states_np = next_states.numpy()
#     dones_np = dones.numpy()

#     for state, action, reward, next_state, done in zip(states_np, actions_np, rewards_np, next_states_np, dones_np):
#         offline_agent.step(state, action, reward, next_state, done)

#     if i_iteration % (NUMITS * 0.01) == 0 and run != None:
#         frames = []
#         num_test_episodes = 10
#         total_test_reward = 0
#         num_test_catastrophes = 0
#         flag = 0

#         for episode in range(num_test_episodes):
#             state, _ = env.reset()
#             for i in range(max_t):
#                 action = offline_agent.act(state)
#                 lander_coordinates = observation_to_pixels(state, viewport_width, viewport_height, scale)

#                 if is_catastrophe(lander_coordinates):
#                     num_test_catastrophes += 1
#                     flag = 1

#                 next_state, reward, terminated, truncated, _ = env.step(action)
#                 done = terminated or truncated

#                 if flag == 1:
#                     reward = -2
#                     flag = 0

#                 frame = render_lines(env.render())
#                 frames.append(frame)

#                 state = next_state
#                 total_test_reward += reward

#                 if done:
#                     break

#         video_frames = np.array(frames)
#         video_frames = np.transpose(video_frames, (0, 3, 1, 2))
#         wandb.log({"Render": wandb.Video(video_frames, fps=15, format="mp4")})

#         average_test_catastrophes = num_test_catastrophes / num_test_episodes
#         average_test_reward = total_test_reward / num_test_episodes
#         wandb.log({"test_catastophe": average_test_catastrophes, "test_score": average_test_reward})


