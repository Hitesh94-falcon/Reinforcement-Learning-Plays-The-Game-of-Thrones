import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


class PadmEnv(gym.Env):
    def __init__(self, grid_size=5):

        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([1, 1])
        self.goal_state = np.array([3, 2])
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=self.grid_size, shape=(2,))
        self.fig, self.ax = plt.subplots()
        plt.show(block=False)

    def reset(self):
        self.agent_state = np.array([1, 1])
        return self.agent_state

    def step(self, action):
        if action == 0 and self.agent_state[1] < self.grid_size:  # up
            self.agent_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0:  # down
            self.agent_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0:  # left
            self.agent_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size:  # right
            self.agent_state[0] += 1

        reward = 0
        done = np.array_equal(self.agent_state, self.goal_state)
        if done:
            reward = 10

        # Now we use Eucledian Distance
        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)

        info = {"Distance to Goal": distance_to_goal}

        return self.agent_state, reward, done, info

    def render(self):
        self.ax.clear()
        self.ax.plot(self.agent_state[0], self.agent_state[1], "ro")
        self.ax.plot(self.goal_state[0], self.goal_state[1], "g+")
        self.ax.set_xlim(-1, self.grid_size)
        self.ax.set_ylim(-1, self.grid_size)
        self.ax.set_aspect("equal")
        plt.pause(0.1)

    def close(self):
        plt.close()


if __name__ == "__main__":
    env = PadmEnv()
    state = env.reset()
    for _ in range(500):
        action = env.action_space.sample()  # action = 0 or 1 or 2 or 3
        state, reward, done, info = env.step(action)
        env.render()
        print(f"State:{state}, Reward:{reward}, Done:{done}, Info:{info}")
        if done:
            print("I reached the goal")
            break
    env.close()
