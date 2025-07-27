# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *



# Function 1: Train Q-learning agent

q_table_dir = "q_tables"


q_table_kingdom1_save_path = os.path.join(q_table_dir, "q_table_kingdom1.npy")
q_table_kingdom2_save_path = os.path.join(q_table_dir, "q_table_kingdom2.npy")
q_table_kingdom3_save_path = os.path.join(q_table_dir, "q_table_kingdom3.npy")
q_table_goal_save_path     = os.path.join(q_table_dir, "q_table_goal.npy")


def train_q_learning(env,
                     no_episodes,
                     epsilon,
                     epsilon_min,
                     epsilon_decay,
                     alpha,
                     gamma,
                     q_table_kingdom1_save_path=q_table_kingdom1_save_path,
                     q_table_kingdom2_save_path=q_table_kingdom2_save_path,
                     q_table_kingdom3_save_path_=q_table_kingdom3_save_path,
                     q_table_goal_save_path=q_table_goal_save_path):


    # Initialize the Q-table:
    q_table_kingdom1 = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    q_table_kingdom2 = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    q_table_kingdom3 = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
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
            Riverlands_captured = False
            Kingslanding_captured =False
            steps = 0
            max_steps = 10000


            #! Step 2: Take actions in the environment until "Done" flag is triggered
            #! -------
            for max_steps in range(max_steps):

                steps += 1

                #! Step 3: Define your Exploration vs. Exploitation
                #! -------

                if np.random.rand() < epsilon:
                    action = env.action_space.sample()  # Explore
                else:
                    if harrenhall_captured==False:      # Exploit
                        action = np.argmax(q_table_kingdom1[state])  
                    elif harrenhall_captured and Riverlands_captured == False:
                        action = np.argmax(q_table_kingdom2[state])
                    elif harrenhall_captured and Riverlands_captured and Kingslanding_captured == False:
                        action = np.argmax(q_table_kingdom3[state])
                    else:
                        action = np.argmax(q_table_goal[state])

                next_state, reward, done, info = env.step(action)
                env.render(render = False)

                next_state = tuple(next_state)
                total_reward += reward

                #! Step 4: Update the Q-values using the Q-value update rule
                #! -------
                if harrenhall_captured == False:
                    q_table_kingdom1[state][action] = q_table_kingdom1[state][action] + alpha * \
                        (reward + gamma * np.max(q_table_kingdom1[next_state]) - q_table_kingdom1[state][action])
                    if info.get("Harrenhall captured"):
                        harrenhall_captured = True
                        print("harenhall captured moving to next kingdom Q table")
                elif Riverlands_captured == False and harrenhall_captured:
                    q_table_kingdom2[state][action] = q_table_kingdom2[state][action] + alpha * \
                        (reward + gamma * np.max(q_table_kingdom2[next_state]) - q_table_kingdom2[state][action])
                    if info.get("Riverlands captured"):
                        Riverlands_captured = True
                        print("harenhall captured moving to next kingdom Q table")
                elif Riverlands_captured and harrenhall_captured and Kingslanding_captured ==False:
                    q_table_kingdom3[state][action] = q_table_kingdom3[state][action] + alpha * \
                        (reward + gamma * np.max(q_table_kingdom3[next_state]) - q_table_kingdom3[state][action])
                    if info.get("KingsLanding captured"):
                        Kingslanding_captured = True
                        print("KingsLanding captured moving to goal Q table")

                elif harrenhall_captured == True and Riverlands_captured == True and Kingslanding_captured == True: 
                    q_table_goal[state][action] = q_table_goal[state][action] + alpha * \
                        (reward + gamma * np.max(q_table_goal[next_state]) - q_table_goal[state][action])

                state = next_state

                #! Step 5: Stop the episode if the agent reaches Goal or Terminal-states
                #! -------
                if done:
                    break


            #! Step 6: Perform epsilon decay
            #! -------
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

            if harrenhall_captured and Riverlands_captured and Kingslanding_captured:
                q_val = q_table_goal[state][action]
            elif harrenhall_captured and Riverlands_captured and not Kingslanding_captured:
                q_val = q_table_kingdom3[state][action]
            elif harrenhall_captured and not Riverlands_captured:
                q_val = q_table_kingdom2[state][action]
            else:
                q_val = q_table_kingdom1[state][action]

            log_episode_result(
                    episode + 1,
                    total_reward,
                    q_val,
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
        np.save(q_table_kingdom1_save_path, q_table_kingdom1)
        np.save(q_table_kingdom2_save_path, q_table_kingdom2)
        np.save(q_table_kingdom3_save_path_, q_table_kingdom3)
        np.save(q_table_goal_save_path,q_table_goal)
        print("Saved the Q-tables.")
    except KeyboardInterrupt:
        np.save(q_table_kingdom1_save_path, q_table_kingdom1)
        np.save(q_table_kingdom2_save_path, q_table_kingdom2)
        np.save(q_table_kingdom3_save_path_, q_table_kingdom3)
        np.save(q_table_goal_save_path,q_table_goal)
        print("keyboard interupt detectd Saved the Q-tables....")

# Function 2: Visualize the Q-tables once Trauned

def visualize_q_table(q_values_path, hell_state_coordinates=[(2,1),(2,4),(5,3),(0,3)], actions=["Down","Up","Left","Right"]):
    try:
        q_table = np.load(q_values_path)

        # Transpose to convert [x][y][action] → [y][x][action] for heatmap plotting
        q_table = q_table.transpose(1, 0, 2)

        _, axes = plt.subplots(1, 4, figsize=(20, 5))

        goal_coordinates = (5,5)  
        hell_state_coordinates = hell_state_coordinates  
        agent_start = (0,0)  
        kingdom = [(3,3),(5,1),(0,5)]

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
            ax.text(ax0 + 0.5, ay0 + 0.5, 'A', color='grey', ha='center', va='center', weight='bold', fontsize=14)
            for (ky, kx) in kingdom:
                mask[ky, kx] = True
                ax.text(kx + 0.5, ky + 0.5, 'K', color='blue', ha='center', va='center', weight='bold', fontsize=14)

            for idx, (hy, hx) in enumerate(hell_state_coordinates):
                ax.text(hx + 0.5, hy + 0.5, 'H', color='red', ha='center', va='center', weight='bold', fontsize=14)

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("No saved Q-table was found.")
