# Imports:
# --------
from my_env import create_env
from Q_learning import *
from gymnasium.wrappers import RecordVideo

# User definitions:
# -----------------
train = True
visualize_results = True


"""
NOTE: Sometimes a fixed initializtion might push the agent to a local minimum.
In this case, it is better to use a random initialization.  
"""
random_initialization = True  

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.999  # Decay rate for exploration
no_episodes = 3_000  # Number of episodes


# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env()

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(env=env,
                     no_episodes=no_episodes,
                     epsilon=epsilon,
                     epsilon_min=epsilon_min,
                     epsilon_decay=epsilon_decay,
                     alpha=learning_rate,
                     gamma=gamma)

if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    visualize_q_table(q_values_path="D:\Thi\Padm\Hail mary\q_value_folder\q_table_kingdom.npy")

    visualize_q_table(q_values_path="D:\Thi\Padm\Hail mary\q_value_folder\q_table_goal.npy")
