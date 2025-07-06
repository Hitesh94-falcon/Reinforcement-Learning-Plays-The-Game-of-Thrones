# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os,csv

csv_path =r"D:\Thi\Padm\Hail mary\logs\log(2).csv"

def log_episode_result(episode, t_rewards, q_values, steps, done, epsilon_):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Result", "Total Reward", "q_values", "steps", "epsilon"])
        writer.writerow([episode, "Success" if done else "Fail", t_rewards, q_values, steps, epsilon_])

# Function 1: Train Q-learning agent
# -----------
def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_kingdom_save_path="D:\Thi\Padm\Hail mary\q_value_folder\q_table_kingdom.npy",
                     q_table_goal_save_path ="D:\Thi\Padm\Hail mary\q_value_folder\q_table_goal.npy"):

    # Initialize the Q-table:
    # -----------------------
    q_table_kingdom = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    q_table_goal = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    # Q-learning algorithm:
    # ---------------------
    #! Step 1: Run the algorithm for fixed number of episodes
    #! -------
    try:
        for episode in range(no_episodes):
            state = env.reset()

            state = tuple(state)
            total_reward = 0
            harrenhall_captured = False
            steps = 0
            max_steps = 10000
            


            #! Step 2: Take actions in the environment until "Done" flag is triggered
            #! -------
            for max_steps in range(max_steps):
                #! Step 3: Define your Exploration vs. Exploitation
                #! -------
                steps += 1

                if np.random.rand() < epsilon:
                    action = env.action_space.sample()  # Explore
                else:
                    if harrenhall_captured==False:
                        action = np.argmax(q_table_kingdom[state])  
                    else:
                        action = np.argmax(q_table_goal[state])

                next_state, reward, done, info = env.step(action)
                # env.render()

                next_state = tuple(next_state)
                total_reward += reward

                #! Step 4: Update the Q-values using the Q-value update rule
                #! -------
                if not harrenhall_captured:
                    q_table_kingdom[state][action] = q_table_kingdom[state][action] + alpha * \
                        (reward + gamma * np.max(q_table_kingdom[next_state]) - q_table_kingdom[state][action])
                    if info.get("Harrenhall captured"):
                        harrenhall_captured = True
                        print("harenhall captured moving to goal Q table")
                        break
                    
                elif harrenhall_captured: 
                    q_table_goal[state][action] = q_table_goal[state][action] + alpha * \
                        (reward + gamma * np.max(q_table_goal[next_state]) - q_table_goal[state][action])

                state = next_state

                #! Step 5: Stop the episode if the agent reaches Goal or Hell-states
                #! -------
                if done:
                    break

            #! Step 6: Perform epsilon decay
            #! -------
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

            log_episode_result(
                    episode + 1,
                    total_reward,
                    q_table_kingdom[state][action] if harrenhall_captured else q_table_goal[state][action],
                    steps=steps,
                    done=done,
                    epsilon_=epsilon
                )
            print(f"In episode {episode + 1}")

        #! Step 7: Close the environment window
        #! -------
        env.close()
        print("Training finished.\n")

        #! Step 8: Save the trained Q-table
        #! -------
        np.save(q_table_kingdom_save_path, q_table_kingdom)
        np.save(q_table_goal_save_path,q_table_goal)
        print("Saved the Q-tables.")
    except KeyboardInterrupt:
        np.save(q_table_kingdom_save_path, q_table_kingdom)
        np.save(q_table_goal_save_path,q_table_goal)
        print("keyboard interupt detectd Saved the Q-tables....")

# Function 2: Visualize the Q-table
# -----------
def visualize_q_table(q_values_path,hell_state_coordinates=[(2,1),(2,4),(4,2),(5,4),(1,7),(0,3),(6,2)], actions=["Up", "Down", "Right", "Left"]):
    try:
        q_table = np.load(q_values_path)

        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        goal_coordinates = (6,6)  # (row, column)
        hell_state_coordinates = hell_state_coordinates  # list of (row, column)
        agent_start = (0,0)  # (row, column)

        for i, action in enumerate(actions):
            ax = axes[i]
            heatmap_data = q_table[:, :, i].copy()

            mask = np.zeros_like(heatmap_data, dtype=bool)
            gy, gx = goal_coordinates  # row, column
            ay0, ax0 = agent_start     # row, column
            mask[gy, gx] = True
            for hy, hx in hell_state_coordinates:
                mask[hy, hx] = True

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
            ax.text(gx + 0.5, gy + 0.5, 'G', color='green', ha='center', va='center', weight='bold', fontsize=14)
            ax.text(ay0 + 0.5, ax0 + 0.5, 'A', color='grey', ha='center', va='center', weight='bold', fontsize=14)

            for idx, (hy, hx) in enumerate(hell_state_coordinates):
                ax.text(hx + 0.5, hy + 0.5, 'H', color='red', ha='center', va='center', weight='bold', fontsize=14)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found.")
