import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from typing import List
import settings as s
import events as e
#from .callbacks import state_to_features

# If GPU is used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

width = s.COLS
height = s.ROWS
num_states = width * height
state_size = num_states  # Define the input size based on your state representation
action_size = len(ACTIONS)  # Define the number of actions


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
q_network = QNetwork(state_size, action_size)
target_q_network = QNetwork(state_size, action_size)
target_q_network.load_state_dict(q_network.state_dict())
target_q_network.eval()


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


# Hyperparameters
LEARNING_RATE = 0.001  # Learning rate for the optimizer
BATCH_SIZE = 64  # Mini-batch size for training
DISCOUNT_FACTOR = 0.99  # Discount factor for future rewards (gamma)
REPLAY_BUFFER_SIZE = 10000  # Size of the replay buffer
TARGET_UPDATE_FREQUENCY = 1000  # Update the target DQN every X steps
TRANSITION_HISTORY_SIZE = 1000  # Size of the transition history

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialize self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    self.optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
def state_to_features(game_state: dict) -> np.array:
    """
    Convert the game state to the input of your model.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    """
    if game_state is None:
        return None

    player_position = game_state["self"][3]  # Player's position as (x, y)
    board = game_state["field"]  # The game board
    max_x, max_y = board.shape[0], board.shape[1]

    # Initialize the state representation as a 3x3 grid centered around the player
    state_representation = np.zeros((3, 3))

    for i in range(-1, 2):
        for j in range(-1, 2):
            x, y = player_position[0] + i, player_position[1] + j

            # Check if the coordinates are within the bounds of the game board
            if 0 <= x < max_x and 0 <= y < max_y:
                cell_content = board[x, y]

                if cell_content == 1:  # Wall
                    state_representation[i + 1, j + 1] = -1
                elif cell_content == 2:  # Crate
                    state_representation[i + 1, j + 1] = 0.5
                elif cell_content == 3:  # Coin
                    state_representation[i + 1, j + 1] = 1

            else:
                state_representation[i + 1, j + 1] = -1

    return state_representation.flatten()

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    if ...:
        events.append(PLACEHOLDER_EVENT)

    self.transitions.append(
        Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),
                   reward_from_events(self, events)))
    if len(self.transitions) >= BATCH_SIZE:
        batch = self.transitions
        self.replay_buffer.push(batch)

    if len(self.replay_buffer) >= BATCH_SIZE:
        self.train_dqn()


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in the final step')
    self.transitions.append(
        Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))
    if len(self.transitions) >= BATCH_SIZE:
        batch = self.transitions
        self.replay_buffer.push(batch)

    if len(self.replay_buffer) >= BATCH_SIZE:
        self.train_dqn()

    if last_game_state["step"] % TARGET_UPDATE_FREQUENCY == 0:
        target_q_network.load_state_dict(q_network.state_dict())
        target_q_network.eval()
    np.save("my-saved-qtable.npy", self.q_table)
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(q_network, file)


def reward_from_events(self, events: List[str]) -> int:
    """
    Modify the rewards your agent gets.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: The events that occurred.
    :return: The modified reward sum.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 5,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -2,
        e.MOVED_LEFT: 10,
        e.MOVED_RIGHT: 10,
        e.MOVED_UP: 10,
        e.MOVED_DOWN: 10,
        PLACEHOLDER_EVENT: -0.1  # The custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def train_dqn(self):
    batch = self.replay_buffer.sample(BATCH_SIZE)
    state_batch, action_batch, next_state_batch, reward_batch = zip(*batch)
