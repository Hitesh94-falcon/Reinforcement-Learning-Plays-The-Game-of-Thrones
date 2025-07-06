# Imports:
# --------
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, csv

def log_episode_result(episode, t_rewards, q_values, steps, done, epsilon_, csv_path):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Result", "Total Reward", "q_values", "steps", "epsilon"])
        writer.writerow([episode, "Success" if done else "Fail", t_rewards, q_values, steps, epsilon_])


def train_q_learning(env, no_episodes, epsilon, epsilon_min, epsilon_decay, alpha, gamma,
                     q_table_kingdom_save_path, q_table_goal_save_path, csv_path):

    q_table_kingdom = np.zeros((env.grid_size, env.grid_size, env.action_space.n))
    q_table_goal = np.zeros((env.grid_size, env.grid_size, env.action_space.n))

    max_steps = 10000

    try:
        for episode in range(no_episodes):
            state = env.reset()
            state = tuple(state)
            total_reward = 0
            done = False
            steps = 0
            kingdoms_captured = False

            for _ in range(max_steps):
                steps += 1

                # Select Q-table
                if not kingdoms_captured:
                    action = np.argmax(q_table_kingdom[state]) if np.random.rand() > epsilon else env.action_space.sample()
                else:
                    action = np.argmax(q_table_goal[state]) if np.random.rand() > epsilon else env.action_space.sample()

                next_state, reward, done, info = env.step(action)
                next_state = tuple(next_state)
                total_reward += reward

                if not kingdoms_captured:
                    q_table_kingdom[state][action] += alpha * (reward + gamma * np.max(q_table_kingdom[next_state]) - q_table_kingdom[state][action])
                    kingdoms_captured = len(env.captured_kingdoms) == len(env.kingdoms)
                else:
                    q_table_goal[state][action] += alpha * (reward + gamma * np.max(q_table_goal[next_state]) - q_table_goal[state][action])

                state = next_state

                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            log_episode_result(
                episode + 1, total_reward,
                q_table_goal[state][action] if kingdoms_captured else q_table_kingdom[state][action],
                steps, done, epsilon, csv_path)

            print(f"Episode {episode + 1}: Total Reward: {total_reward}")

        env.close()
        np.save(q_table_kingdom_save_path, q_table_kingdom)
        np.save(q_table_goal_save_path, q_table_goal)
        print("Saved both Q-tables.")

    except KeyboardInterrupt:
        np.save(q_table_kingdom_save_path, q_table_kingdom)
        np.save(q_table_goal_save_path, q_table_goal)
        print("Training interrupted. Q-tables saved.")


def visualize_q_table(env, q_values_paths, actions=["Up", "Down", "Right", "Left"]):
    phase_labels = ["Kingdom Phase", "Goal Phase"]  

    for idx, path in enumerate(q_values_paths):
        try:
            q_table = np.load(path)
            _, axes = plt.subplots(1, 4, figsize=(20, 5))

            goal = env.goal_state
            obstacles = sum(env.obstacles.values(), [])
            agent = env.agent_state

            for i, action in enumerate(actions):
                ax = axes[i]
                heatmap_data = q_table[:, :, i].copy()
                mask = np.zeros_like(heatmap_data, dtype=bool)

                if goal is not None:
                    mask[tuple(goal)] = True
                for obs in obstacles:
                    mask[tuple(obs)] = True

                sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                            ax=ax, cbar=False, mask=mask, annot_kws={"size": 9})

                if goal is not None:
                    ax.text(goal[1] + 0.5, goal[0] + 0.5, 'G', color='green',
                            ha='center', va='center', weight='bold', fontsize=14)
                for h in obstacles:
                    ax.text(h[1] + 0.5, h[0] + 0.5, 'H', color='red',
                            ha='center', va='center', weight='bold', fontsize=14)
                ax.text(agent[1] + 0.5, agent[0] + 0.5, 'A', color='blue',
                        ha='center', va='center', weight='bold', fontsize=14)

                ax.set_title(f'{phase_labels[idx]} - {action}')

            plt.tight_layout()
            plt.show()
            input("Press Enter to visualize the next Q-table...")

        except FileNotFoundError:
            print(f"Q-table not found: {path}")
