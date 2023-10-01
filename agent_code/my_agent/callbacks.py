import hashlib
import os
import pickle
import random

import numpy as np
from .train import state_to_features
from agent_code.my_agent.train import num_states

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    '''if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        weights = np.random.rand(len(ACTIONS))
        self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.model = pickle.load(file)'''

    if self.train or not os.path.isfile("my-saved-qtable.npy"):
        self.logger.info("Setting up Q-table from scratch.")
        # Initialize the Q-table with zeros
        self.q_table = np.zeros((len(ACTIONS), num_states))  # Define the shape based on your state representation
    else:
        self.logger.info("Loading Q-table from saved state.")
        # Load the Q-table from a saved file
        self.q_table = np.load("my-saved-qtable.npy")


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    random_prob = .1
    if self.train and random.random() < random_prob:
        self.logger.debug("Choosing action purely at random.")
        # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS)

    self.logger.debug("Querying model for action.")
    state_features = state_to_features(game_state)

    def encode_state(state_features):
        # Assuming state_features is a 1D NumPy array
        state_str = ''.join(map(str, state_features))
        state_hash = int(hashlib.md5(state_str.encode()).hexdigest(), 16)
        return state_hash
    if state_features is not None:

        #encode_state(state_features)
        #q_values = self.q_table[state_features]
        normalized_state_features = (state_features + 1) / 2  # Normalize values to [0, 1]
        max_index = len(self.q_table) - 1  # Maximum integer index
        indices = (normalized_state_features * max_index).astype(int)  # Map to integer indices
        q_values = self.q_table[indices]

        action = ACTIONS[np.argmax(q_values)]
    else:
        action = np.random.choice(ACTIONS)

    return action
    #return np.random.choice(ACTIONS, p=self.model)

"""
def state_to_features(game_state: dict) -> np.array:
    
    Convert the game state to the input of your model.

    :param game_state: A dictionary describing the current game board.
    :return: np.array
    
    if game_state is None:
        return None

    player_x, player_y = game_state["self"]  # Player's position
    board = game_state["field"]  # The game board

    # Initialize the state representation as a 3x3 grid centered around the player
    state_representation = np.zeros((3, 3))

    # Populate the state representation with information about the surrounding cells
    for i in range(-1, 2):
        for j in range(-1, 2):
            x, y = player_x + i, player_y + j

            # Check if the coordinates are within the bounds of the game board
            if 0 <= x < board.shape[0] and 0 <= y < board.shape[1]:
                cell_content = board[x, y]

                # Map cell content to values in the state representation
                if cell_content == 1:  # Wall
                    state_representation[i + 1, j + 1] = -1
                elif cell_content == 2:  # Crate
                    state_representation[i + 1, j + 1] = 0.5
                elif cell_content == 3:  # Coin
                    state_representation[i + 1, j + 1] = 1

            # If the coordinates are out of bounds, mark them as obstacles
            else:
                state_representation[i + 1, j + 1] = -1

    # Flatten the 3x3 grid into a 1D array
    return state_representation.flatten()
"""