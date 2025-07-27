import os,csv

csv_path =r"D:\Thi\Padm\experimentation\logs\log(8).csv"

def log_episode_result(episode, t_rewards, q_values_, steps, done, epsilon_):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Result", "Total Reward", "q_values", "steps", "epsilon"])
        writer.writerow([episode, "Success" if done else "Fail", t_rewards, q_values_, steps, epsilon_])