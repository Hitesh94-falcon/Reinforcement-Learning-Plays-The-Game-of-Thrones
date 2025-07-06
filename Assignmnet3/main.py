# NOTE: Code adapted from MinimalRL (URL: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)

# Imports:
# --------
import torch
import gymnasium as gym
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from env import create_env

device = torch.device("cuda")

# User definitions:
# -----------------
train_dqn = True
test_dqn = True
render = False

#! Define env attributes (environment specific)
dim_actions = 4
dim_states = 2


# Hyperparameters:
# ----------------
learning_rate = 0.005
gamma = 0.98
buffer_limit = 50_000
batch_size = 32
num_episodes = 10_000
max_steps = 10_00

csv_path =r"D:\Thi\Padm\Hail mary\logs\log(6).csv"

def log_episode_result(episode, episode_reward, q_values, steps, done, epsilon_):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Episode", "Result", "episode_reward", "q_values", "steps", "epsilon"])
        writer.writerow([episode, "Success" if done else "Fail", episode_reward, q_values, steps, epsilon_])

# Main:
# -----
try:
    if train_dqn:
        env=create_env()

        #! Initialize the Q Net and the Q Target Net
        # q_net = Qnet(dim_actions=dim_actions, 
        #              dim_states=dim_states)
        # q_target = Qnet(dim_actions=dim_actions, 
        #                 dim_states=dim_states)
        q_net = Qnet(dim_actions, dim_states).to(device)
        q_target = Qnet(dim_actions, dim_states).to(device)
        
        q_target.load_state_dict(q_net.state_dict())

        #! Initialize the Replay Buffer
        memory = ReplayBuffer(buffer_limit=buffer_limit)

        print_interval = 10
        episode_reward = 0.0
        optimizer = optim.Adam(q_net.parameters(),
                            lr=learning_rate)

        rewards = []

        for n_epi in range(num_episodes):
            #! Epsilon decay (Please come up with your own logic)
            epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)
                        )  # ! Linear annealing from 8% to 1%

            s, _ = env.reset()
            done = False
            steps = 0
            #! Define maximum steps per episode, here 1,000
            for _ in range(max_steps):
                #! Choose an action (Exploration vs. Exploitation)
                steps += 1
                s_prime = torch.from_numpy(s).float().to(device)
                a = q_net.sample_action(s_prime, epsilon)
                s_prime, r, done, _, _ = env.step(a)

                s_prime_tensor = torch.from_numpy(s_prime).float().to(device)

                done_mask = 0.0 if done else 1.0

                #! Save the trajectories
                memory.put((s, a, r, s_prime, done_mask))
                s = s_prime

                episode_reward += r

                if done:
                    break

            if memory.size() > 2000:
                train(q_net, q_target, memory, optimizer, batch_size, gamma)

            if n_epi % print_interval == 0 and n_epi != 0:
                q_target.load_state_dict(q_net.state_dict())
                print(
                    f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

            rewards.append(episode_reward)
            episode_reward = 0.0

            #! Define a stopping condition for the game:
            if rewards[-10:] == [max_steps]*10:
                break

        env.close()

        #! Save the trained Q-net
        torch.save(q_net.state_dict(), "dqn.pth")

        #! Plot the training curve
        plt.plot(rewards, label='Reward per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        plt.legend()
        plt.savefig("training_curve.png")
        plt.show()

except KeyboardInterrupt:
    torch.save(q_net.state_dict(), "dqn.pth")

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    
    env = create_env()

    dqn = Qnet(dim_actions=dim_actions, 
               dim_states=dim_states)
    dqn.load_state_dict(torch.load("dqn.pth"))

    for _ in range(100):
        s, _ = env.reset()
        episode_reward = 0

        for _ in range(max_steps):
            #! Completely exploit while testing
            action = dqn(torch.from_numpy(s).float())
            s_prime, reward, done, _, _ = env.step(action.argmax().item())
            s = s_prime

            episode_reward += reward

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()
