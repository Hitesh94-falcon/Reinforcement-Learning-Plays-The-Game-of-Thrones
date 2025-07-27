# Imports:
# --------
from my_env import create_env
from Q_learning import *


# User definitions:
# -----------------
train = True
visualize_results = True
 

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.999  # Decay rate for exploration
no_episodes = 3000  # Number of episodes


# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env()

    # Train the Q-learning agent:
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
    visualize_q_table(q_values_path=q_table_kingdom1_save_path)
    visualize_q_table(q_values_path=q_table_kingdom2_save_path)
    visualize_q_table(q_values_path=q_table_kingdom3_save_path)
    visualize_q_table(q_values_path=q_table_goal_save_path)

