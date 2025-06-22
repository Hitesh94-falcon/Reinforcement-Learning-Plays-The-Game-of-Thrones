# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv,os

csv_path = r"D:\Thi\Padm\Assignment 2\intermeditaeqtables\log(rand_inti).csv"

def log_episode_result(episode,t_rewards):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Result","Total Reward"])
        writer.writerow([episode, "Success", t_rewards])


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy",
                     ):

    # Initialize the Q-table:
    # -----------------------
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    try:
        for episode in range(no_episodes):
            state = env.reset()

            state = tuple(state)
            total_reward = 0

            #! Step 2: Take actions in the environment until "Done" flag is triggered
            #! -------
            while True:
                #! Step 3: Define your Exploration vs. Exploitation
                #! -------
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()  # Explore
                else:
                    action = np.argmax(q_table[state])  # Exploit

                next_state, reward, done, _ = env.step(action)

                next_state = tuple(next_state)
                total_reward += reward

                #! Step 4: Update the Q-values using the Q-value update rule
                #! -------
                q_table[state][action] = q_table[state][action] + alpha * \
                    (reward + gamma *
                    np.max(q_table[next_state]) - q_table[state][action])

                state = next_state

                #! Step 5: Stop the episode if the agent reaches Goal or Hell-states
                #! -------
                if done:
                    break

            #! Step 6: Perform epsilon decay
            #! -------
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

            print(f"in episode{episode+1}")
            if (episode + 1) % 100 == 0:
                np.save(f"D:\Thi\Padm\Assignment 2\intermeditaeqtables\q_table_ep{episode+1}.npy", q_table)
                print(f"Checkpoint saved: q_table_ep{episode+1}.npy")
            
            if done:
                log_episode_result(episode + 1,total_reward)
            else:
                log_episode_result(episode+1,total_reward)

        #! Step 7: Close the environment window
        #! -------
        env.close()
        print("Training finished.\n")

        #! Step 8: Save the trained Q-table
        #! -------
        np.save(q_table_save_path, q_table)
        print("Saved the final Q-table.")

    except KeyboardInterrupt:
        print("keyboard interupt detected,saving Q_table")
        np.save(q_table_save_path,q_table)

# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(hell_state_coordinates=[(0, 1), (0, 8), (8, 1), (4, 0), (1, 0), (8, 4), (8, 8), (4, 3), (2, 5)],
                      goal_coordinates=(5,7),
                      actions=["Up", "Down", "Right", "Left"],
                      q_values_path="q_table.npy"):

    # Load the Q-table:
    # -----------------
    # try:
    #     q_table = np.load(q_values_path)

    #     # Create subplots for each action:
    #     # --------------------------------
    #     _, axes = plt.subplots(1, 4, figsize=(20, 5))

    #     for i, action in enumerate(actions):
    #         ax = axes[i]
    #         heatmap_data = q_table[:, :, i].T 

    #         # Mask the goal state's Q-value for visualization:
    #         # ------------------------------------------------
    #         mask = np.zeros_like(heatmap_data, dtype=bool)
    #         mask[goal_coordinates] = True
    #         for hx, hy in hell_state_coordinates:
    #             mask[hx, hy] = True

    #         # sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
    #         #             ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

    #         sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
    #         ax=ax, cbar=False, mask=mask.T, annot_kws={"size": 9},
    #         square=True, linewidths=0.1, linecolor='gray')

    #         ax.invert_yaxis() 

    #         # Denote Goal and Hell states:
    #         # ----------------------------
    #         ax.text(goal_coordinates[1] + 0.5, goal_coordinates[0] + 0.5, 'G', color='green',
    #                 ha='center', va='center', weight='bold', fontsize=14)
    #         if mask[hx,hy]:
    #             for hx, hy in hell_state_coordinates:
    #                 ax.text(hy + 0.5, hx + 0.5, 'H', color='red',
    #                 ha='center', va='center', weight='bold', fontsize=14)
    #         ax.text(hell_state_coordinates[1][1] + 0.5, hell_state_coordinates[1][0] + 0.5, 'H', color='red',
    #                 ha='center', va='center', weight='bold', fontsize=14)

    #         ax.set_title(f'Action: {action}')

    #     plt.tight_layout()
    #     plt.show()

    # except FileNotFoundError:
    #     print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")
    try:
        q_table = np.load(q_values_path)  # Shape: (rows, cols, actions)

        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()  

            # Create mask
            mask = np.zeros_like(heatmap_data, dtype=bool)
            gx, gy = goal_coordinates
            initial_agent_coordinates = (4,1)
            Ax, Ay = initial_agent_coordinates
            mask[gx, gy] = True
            for hx, hy in hell_state_coordinates:
                mask[hx, hy] = True

            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".2f",
                cmap="viridis",
                ax=ax,
                cbar=False,
                mask=mask,
                annot_kws={"size": 9},
                square=True,
                linewidths=0.1,
                linecolor='gray'
            )

            # ax.invert_yaxis()  # This makes (0, 0) at top-left to match Pygame grid
            ax.set_title(f'Action: {action}')

            # Plot 'G' for goal
            ax.text(gy + 0.5, gx + 0.5, 'G', color='green',
                    ha='center', va='center', weight='bold', fontsize=14)
            
            ax.text(Ay + 0.5, Ax + 0.5, 'A', color='grey',
                    ha='center', va='center', weight='bold', fontsize=14)

            # Plot 'H' for all hell states
            for hx, hy in hell_state_coordinates:
                ax.text(hy + 0.5, hx + 0.5, 'H', color='red',
                        ha='center', va='center', weight='bold', fontsize=14)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found. Please train the Q-learning agent first or check your path.")

