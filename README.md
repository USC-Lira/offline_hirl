### Setup and Requirements
Prerequisites:
- Python 3.6 or later
- PyTorch
- gymnasium
- numpy
- wandb (for logging and visualization)
- PIL (for image handling)
```
pip install torch gymnasium numpy wandb pillow
```

## Mountain Car

### Overview
1. **mcdqn.py**: Implements a Deep Q-Network for the Mountain Car environment.
2. **mc_offlinerl.py**: Collects experiences using a provided policy and then trains a new policy offline with these experiences.

### Usage
#### Training the DQN Agent
To train the DQN agent on the Mountain Car environment, run:
```
python3 mcdqn.py --name experiment_name --numeps 3000 --seed 0 --penalty 2 --wandb
```
Arguments:
- --name: Set the name of the experiment.
- --numeps: Sets the number of episodes for training.
- --seed: Sets the seed.
- --penalty: Sets the penalty for catastrophic actions (set to negative in the implementation).
- --addha: Include this flag to add human actions to the training.
- --wandb: Include this flag to log the training process to Weights & Biases.

#### Collecting Experiences and Offline RL
To collect experiences using an optimal policy and train a new policy offline, run:
```
python mc_offlinerl.py --name experiment_name --numeps 1000 --numits 3000000 --seed 0 --penalty 2 --addha --wandb
```
Arguments:
- --name: Set the name of the experiment.
- --numeps: Number of episodes for collecting experiences.
- --numits: Number of iterations for offline training.
- --seed: Sets the seed.
- --penalty: Sets the penalty for catastrophic actions (set to negative in the implementation).
- --addha: Include this flag to add human actions to the buffer.
- --wandb: Include this flag to log the process to Weights & Biases.

Currently, you will need to modify the path of the `.pth` file in this line of code to load the optimal policy:
```
optimal_policy.load_state_dict(torch.load('/home/jaiv/offlinerl_dqn/checkpoints/mc_hirl_v1_s0/mc_dqn_2500.pth', map_location=device))
```
