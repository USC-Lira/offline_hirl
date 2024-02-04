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
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
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
    n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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
        state, _ = env.reset()
        score = 0
        num_blocker = 0
        num_catastrophe = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            robot_action = action

            lander_coordinates = observation_to_pixels(state, viewport_width, viewport_height, scale)
            
            if is_blocker(lander_coordinates):
                flag = 1
                action = 1
                num_blocker += 1

            if is_catastrophe(lander_coordinates):
                num_catastrophe += 1
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if flag == 1:
                reward = -2
                flag = 0

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
                        lander_coordinates = observation_to_pixels(state, viewport_width, viewport_height, scale)

                        if is_catastrophe(lander_coordinates):
                            num_test_catastrophes += 1
                            flag = 1

                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated

                        if flag == 1:
                            reward = -2
                            flag = 0

                        frame = render_lines(env.render())
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
            savename = "./checkpoints/" + str(EXPERIMENT_NAME) + "/dqn_" + str(i_episode) + ".pth"
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), savename)
            savenumber += 1
        
    return scores

def observation_to_pixels(observation, viewport_width, viewport_height, scale):
    # Scale factors as per the environment's specification
    scale_factors = {'x': 10, 'y': 6.666}

    # Extract position from the observation and apply scale factors
    x_pos_scaled = observation[0] * scale_factors['x']
    y_pos_scaled = observation[1] * scale_factors['y']

    # Convert x position: Center of the screen is 0 in the environment's coordinates
    x_pixel = (x_pos_scaled + viewport_width / scale / 2) * scale

    # Convert y position: Invert y axis for rendering (screen's 0,0 is at top-left)
    y_pixel = viewport_height - (y_pos_scaled + viewport_height / scale / 4) * scale

    return x_pixel, y_pixel

def is_blocker(lander_coordinates):
    # Coordinates of the blue box
    box_top_left = (360, 120)
    box_bottom_right = (600, 400)

    # Extract the lander's position
    lander_x, lander_y = lander_coordinates

    # Check if the lander's position is within the blocker box boundaries
    if box_top_left[0] <= lander_x <= box_bottom_right[0] and box_top_left[1] <= lander_y <= box_bottom_right[1]:
        return True
    else:
        return False
    
def is_catastrophe(lander_coordinates):
    # Coordinates of the red box
    red_box_top_left = (395, 155)
    red_box_bottom_right = (600, 400)

    # Extract the lander's position
    lander_x, lander_y = lander_coordinates

    # Check if the lander's position is within the catastrophe box boundaries
    if red_box_top_left[0] <= lander_x <= red_box_bottom_right[0] and red_box_top_left[1] <= lander_y <= red_box_bottom_right[1]:
        return True
    else:
        return False

def render_lines(frame):
    if frame is not None:
        blue_color = (0, 0, 255)
        red_color = (255, 0, 0)
        green_color = (0, 255, 0)

        line_thickness = 1

        # BLUE BOX (BLOCKER ZONE):
        # Define the coordinates for the horizontal blue line
        horiz_start = (360, 120)
        horiz_end = (600, 120)

        frame[horiz_start[1]:horiz_start[1] + line_thickness, horiz_start[0]:horiz_end[0]] = blue_color

        # Define the coordinates for the vertical blue line
        vert_start = (360, 120)
        vert_end = (360, 400)

        frame[vert_start[1]:vert_end[1], vert_start[0]:vert_start[0] + line_thickness] = blue_color

        # RED BOX (CATASTROPHE ZONE):
        # Define the coordinates for the horizontal red line
        horiz_start = (395, 155)
        horiz_end = (600, 155)

        frame[horiz_start[1]:horiz_start[1] + line_thickness, horiz_start[0]:horiz_end[0]] = red_color

        # Define the coordinates for the vertical red line
        vert_start = (395, 155)
        vert_end = (395, 400)

        frame[vert_start[1]:vert_end[1], vert_start[0]:vert_start[0] + line_thickness] = red_color
    
    return frame

# Handle Argument Parsing:
parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="set the name of the experiment")
parser.add_argument("--numeps", type=int, help="sets number of episodes")
parser.add_argument("--seed", type=int, help="sets seed")
parser.add_argument("--wandb", help="sets wandb to be true", action="store_true")
args = parser.parse_args()

BUFFER_SIZE = int(1e5)          # replay buffer size
BATCH_SIZE = 64                 # minibatch size
GAMMA = 0.99                    # discount factor
TAU = 1e-3                      # for soft update of target parameters
LR = 5e-4                       # learning rate
UPDATE_EVERY = 4                # how often to update the network

WANDB = args.wandb              # log using wandb
EXPERIMENT_NAME = args.name     # set experiment name
SEED = args.seed                # seed
NUMEPS = args.numeps            # number of episodes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Handle wandb:
wandb_hyperparameters = {
    'env_name': 'LunarLander-v2',
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
    'exp_ver': 'v7',
}

run = wandb.init(project="modified_hirl", name=EXPERIMENT_NAME, config=wandb_hyperparameters) if WANDB else None

viewport_width = 600  # VIEWPORT_W
viewport_height = 400  # VIEWPORT_H
scale = 30.0  # SCALE

os.makedirs("checkpoints/" + str(EXPERIMENT_NAME), exist_ok=True)
env = gym.make("LunarLander-v2", render_mode="rgb_array")
agent = Agent(state_size=8, action_size=4, seed=SEED)
train(agent, n_episodes=NUMEPS)