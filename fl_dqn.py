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
from PIL import Image, ImageDraw, ImageFont

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

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state_one_hot = np.zeros(self.state_size)
        state_one_hot[state] = 1
        state = torch.from_numpy(state_one_hot).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Flatten the states and next_states tensors and convert them to integer type
        states_np = states.squeeze().long().cpu().numpy()
        next_states_np = next_states.squeeze().long().cpu().numpy()

        # Create one-hot encoded numpy arrays
        states_one_hot_np = np.eye(self.state_size)[states_np]
        next_states_one_hot_np = np.eye(self.state_size)[next_states_np]

        # Convert one-hot encoded numpy arrays back to PyTorch tensors
        states_one_hot = torch.from_numpy(states_one_hot_np).float().to(device)
        next_states_one_hot = torch.from_numpy(next_states_one_hot_np).float().to(device)

        # Continue with the existing logic for Q-learning
        Q_targets_next = self.qnetwork_target(next_states_one_hot).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states_one_hot).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        if run is not None:
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
        self.memory = deque(maxlen=buffer_size)
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
    
def train(agent,
    n_episodes=1000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.999):
    """Deep Q-Learning.

    Params
    ======
        agent: agent to train
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    # writer = SummaryWriter(log_dir='dqn_runs/')
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    savenumber = 0
    
    flag = 0
    num_blocker = 0
    num_catastrophe = 0

    for i_episode in range(1, n_episodes+1):
        if not ADD_HA: # not using human actions
            state, _ = env.reset()
            score = 0
            num_blocker = 0
            num_catastrophe = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                robot_action = action
                
                if detect_catastrophe(state, action, env):
                    flag = 1
                    action = 2
                    num_blocker += 1

                if is_catastrophe(state):
                    num_catastrophe += 1
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if flag == 1:
                    reward = penalty
                    flag = 0

                agent.step(state, robot_action, reward, next_state, done)
                
                state = next_state
                score += reward

                if done:
                    break
        else:
            state, _ = env.reset()
            score = 0
            num_blocker = 0
            num_catastrophe = 0
            for t in range(max_t):
                action = agent.act(state, eps)
                robot_action = action
                
                if detect_catastrophe(state, action, env):
                    flag = 1
                    action = 2
                    num_blocker += 1

                if is_catastrophe(state):
                    num_catastrophe += 1
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if flag == 1:
                    agent.step(state, robot_action, penalty, next_state, done)
                    agent.step(state, action, reward, next_state, done)
                    reward = penalty
                    flag = 0
                else:
                    agent.step(state, robot_action, reward, next_state, done)
                
                state = next_state
                score += reward

                if done:
                    break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        
        if run != None:
            wandb.log({"num_episodes": i_episode, "score": score, "average_score": np.mean(scores_window), "epsilon": eps, "train_catastrophe": num_catastrophe, "train_blocker": num_blocker})
            
            if i_episode % 100 == 0:
                frames = []
                num_test_episodes = 10
                total_test_reward = 0
                num_test_catastrophes = 0
                flag = 0

                for episode in range(num_test_episodes):
                    state, _ = env.reset()
                    for i in range(max_t):
                        action = agent.act(state, 0.01)

                        if is_catastrophe(state):
                            num_test_catastrophes += 1
                            flag = 1
                            frame = frame_write(env.render())
                        else:
                            frame = env.render()

                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        if flag == 1:
                            reward = -2
                            flag = 0

                        frames.append(frame)

                        state = next_state
                        total_test_reward += reward

                        if done:
                            break

                video_frames = np.array(frames)
                video_frames = np.transpose(video_frames, (0, 3, 1, 2))
                wandb.log({"Render": wandb.Video(video_frames, fps=15, format="mp4")})

                average_test_catastrophes = num_test_catastrophes / num_test_episodes
                average_test_reward = total_test_reward / num_test_episodes
                wandb.log({"test_catastophe": average_test_catastrophes, "test_score": average_test_reward})
              
        if i_episode % 100 == 0:
            savename = "./checkpoints/" + str(EXPERIMENT_NAME) + "/mc_dqn_" + str(i_episode) + ".pth"
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), savename)
            savenumber += 1
        
    return scores

def detect_catastrophe(state, action, env):
    # Define the action mappings
    # In an 8x8 grid: 0: Left, 1: Down, 2: Right, 3: Up
    action_effects = {0: -1, 1: 8, 2: 1, 3: -8}
    
    # Calculate the next state
    next_state = state + action_effects[action]

    # Check if the next state is out of bounds
    if next_state < 0 or next_state >= env.observation_space.n:
        return False

    # Check if the next state wraps around to the next row
    if action == 0 and state % 8 == 0:  # Adjusted for 8x8 grid
        return False
    if action == 2 and (state + 1) % 8 == 0:  # Adjusted for 8x8 grid
        return False

    # Check if the next state is in a catastrophe zone
    catastrophe_zones = [4, 5, 6, 7, 13, 14, 15]  # Adjusted zone indices
    return next_state in catastrophe_zones

def is_catastrophe(state):  # This function is now redundant given the above logic
    catastrophe_zones = [4, 5, 6, 7, 13, 14, 15]
    return int(state) in catastrophe_zones

def frame_write(frame):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.text((100, 100), "catastrophe", fill=(255, 0, 0)) #, font=font)
    frame_with_text = np.array(img)
    return frame_with_text

# Handle Argument Parsing:
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="set the name of the experiment")
parser.add_argument("--numeps", type=int, help="sets number of episodes")
parser.add_argument("--seed", type=int, help="sets seed")
parser.add_argument("--penalty", type=int, help="sets penalty")
parser.add_argument("--addha", help="sets human actions to be true", action="store_true")
parser.add_argument("--wandb", help="sets wandb to be true", action="store_true")
args = parser.parse_args()

BUFFER_SIZE = int(1e5)          # replay buffer size
BATCH_SIZE = 64                 # minibatch size
GAMMA = 0.99                    # discount factor
TAU = 1e-3                      # for soft update of target parameters
LR = 1e-3                       # learning rate
UPDATE_EVERY = 4                # how often to update the network

WANDB = args.wandb              # log using wandb
EXPERIMENT_NAME = args.name     # set experiment name
SEED = args.seed                # seed
PENALTY = args.penalty          # penalty (positive in argument, set to be negative below)
ADD_HA = args.addha             # sets if human actions are in the buffer
NUMEPS = args.numeps            # number of episodes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Handle wandb:
wandb_hyperparameters = {
    'env_name': 'FrozenLake-v1',
    'n_episodes': NUMEPS,
    'experiment_name': EXPERIMENT_NAME,
    'dualbuf': True,
    'buffer_size': BUFFER_SIZE,
    'batch_size': BATCH_SIZE,
    'gamma': GAMMA,
    'tau': TAU,
    'learning_rate': LR,
    'update_every': UPDATE_EVERY,
    'penalty': -2,
    'seed': SEED,
    'add_ha': ADD_HA,
    'exp_ver': 'v7',
}

penalty = float(-PENALTY)

run = wandb.init(project="modified_hirl", name=EXPERIMENT_NAME, config=wandb_hyperparameters) if WANDB else None

os.makedirs("checkpoints/" + str(EXPERIMENT_NAME), exist_ok=True)
env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=False, render_mode="rgb_array")
# env = gym.make("MountainCar-v0", render_mode="rgb_array")
agent = Agent(state_size=64, action_size=4, seed=SEED)
train(agent, n_episodes=NUMEPS)