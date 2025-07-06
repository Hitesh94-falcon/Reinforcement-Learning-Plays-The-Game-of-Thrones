# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv,os

csv_path = r"D:\Thi\Padm\Assignment 2\intermeditaeqtables\log(11).csv"

def log_episode_result(episode,t_rewards,q_values,hurray,epsilon_,steps):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Result","Total Reward","q_values", "epsilon", "max_step"])
        writer.writerow([episode, "Success" if hurray else "Fail", t_rewards, q_values, epsilon_, steps])


# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_save_path="q_table.npy"):

    # Initialize the Q-table:
    # -----------------------
    q_table = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Q-learning algorithm:
    # ---------------------
    #! Run the algorithm for fixed number of episodes
    #! -------
    max_steps = 20000
    try:
        for episode in range(no_episodes):
            state = env.reset()
            state = tuple(state)
            total_reward = 0
            hurray = False
            steps = 0
            for max_step in range(max_steps):

                # Step 3: Exploration vs. Exploitation
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()  # Explore
                else:
                    action = np.argmax(q_table[state])  # Exploit

                next_state, reward, done, _ = env.step(action)
                env.render()

                next_state = tuple(next_state)
                total_reward += reward

                # Step 4: Q-learning update
                q_table[state][action] = q_table[state][action] + alpha * (
                    reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
                )

                state = next_state
                steps += 1

                if done:
                    if env.reached_goal:
                        hurray = True
                    break  

            # Epsilon decay after each episode
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            # Logging
            log_episode_result(
                    episode + 1,
                    total_reward,
                    q_table[state][action],
                    hurray=hurray,
                    epsilon_=epsilon,
                    steps=steps
                )

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

            if (episode + 1) % 500 == 0:
                save_path = f"D:\\Thi\\Padm\\Assignment 2\\intermeditaeqtables\\q_table_ep{episode + 1}_randiniti.npy"
                np.save(save_path, q_table)
                print(f"Checkpoint saved: q_table_ep{episode + 1}_randiniti.npy")

        #! Close the environment window
        #! -------
        env.close()
        print("Training finished.\n")

        #! Save the trained Q-table
        #! -------
        np.save(q_table_save_path, q_table)
        print("Saved the final Q-table.")

    except KeyboardInterrupt:
        print("keyboard interupt detected,saving Q_table")
        np.save(q_table_save_path,q_table)

# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(env, q_values_path, actions=["Up", "Down", "Right", "Left"]):
    try:
        q_table = np.load(q_values_path)

        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        goal_coordinates = tuple(env.goal_state)
        hell_state_coordinates = [tuple(obs) for obs in sum(env.obstacles.values(), [])]
        agent_start = tuple(env.agent_state)

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            mask = np.zeros_like(heatmap_data, dtype=bool)
            gx, gy = goal_coordinates
            ax0, ay0 = agent_start
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

            ax.set_title(f'Action: {action}')
            ax.text(gy + 0.5, gx + 0.5, 'G', color='green', ha='center', va='center', weight='bold', fontsize=14)
            ax.text(ay0 + 0.5, ax0 + 0.5, 'A', color='grey', ha='center', va='center', weight='bold', fontsize=14)

            for hx, hy in hell_state_coordinates:
                ax.text(hy + 0.5, hx + 0.5, 'H', color='red', ha='center', va='center', weight='bold', fontsize=14)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found.")

