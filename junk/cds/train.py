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
# random_initialization = True  # If True, the Q-table will be initialized randomly

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.1  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 5_000  # Number of episodes

# goal_coordinates = (7,5)

# # Define all hell state coordinates as a tuple within a list
# hell_state_coordinates = [(1, 0), (8, 0), (1, 8), (0, 4), (0, 1), (4, 8), (8, 8),(3, 4), (5, 2)]


# Execute:
# --------
if train:
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env()

    # env_rec = RecordVideo(
    # env,
    # video_folder= r"D:\Thi\Padm\viedos",
    # episode_trigger=lambda ep: ep % 500 == 0,
    # name_prefix="qlearning"
    # )

    # Train a Q-learning agent:
    # -------------------------
    train_q_learning(
        env=env,
        no_episodes=no_episodes,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        alpha=learning_rate,
        gamma=gamma,
        q_table_kingdom_save_path=r"D:\Thi\Padm\Hail mary\q_value_folder\q_table_kingdom.npy",
        q_table_goal_save_path=r"D:\Thi\Padm\Hail mary\q_value_folder\q_table_goal.npy",
        csv_path=r"D:\Thi\Padm\Hail mary\logs\log.csv"
    )


if visualize_results:
    # Visualize the Q-table:
    # ----------------------
    visualize_q_table(env=env, q_values_folder="D:\Thi\Padm\Hail mary\q_value_folder")
