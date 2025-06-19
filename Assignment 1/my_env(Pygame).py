import gymnasium as gym
import numpy as np
import pygame


class PadmEnv(gym.Env):
    def __init__(self, grid_size=10, tile_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.agent_state = np.array([1, 4])
        self.goal_state = np.array([7, 5])
        self.action_space = gym.spaces.Discrete(4)

        self.obstacles = {
            "Dragon": [(0, 1), (0, 8), (8, 1)],
            "Night-walker": [(1,0), (8, 4)],
            "Army": [(5,4), (4, 3), (2, 5)]
        }

        self.rewards = {
            "Dragon-eggs": [(6, 2), (6, 9), (9, 4)],
            "Kingdoms": [(3, 3), (2, 6), (5, 6), (8, 7)]
        }

        self.kingdoms = ["Harrenhall", "River-Lands", "Kings-Landing", "storm-lands"]
        self.observation_space = gym.spaces.Box(low=0, high=grid_size, shape=(2,), dtype=np.int32)

        # Initialize Pygame
        pygame.init()
        self.screen_size = self.grid_size * self.tile_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("PadmEnv")

        # Load images
        self.images = {
            "Dragon": pygame.image.load(r"D:\Thi\Padm\Images\obstacles\dragon_balerion.png"),
            "Night-walker": pygame.image.load(r"D:\Thi\Padm\Images\obstacles\nightwalker.png"),
            "Army": pygame.image.load(r"D:\Thi\Padm\Images\obstacles\Army.png"),
            "Daenerys-Targaryen": pygame.image.load(r"D:\Thi\Padm\Images\states\Rahnerya.png"),
            "Iron-throne": pygame.image.load(r"D:\Thi\Padm\Images\states\iron_throne.png"),
            "Dragon-eggs": pygame.image.load(r"D:\Thi\Padm\Images\rewards\Dragon-eggs.png"),
            "kingdoms": {
                "Harrenhall": pygame.image.load(r"D:\Thi\Padm\Images\rewards\Kingdoms\Harrenhall.jpeg"),
                "River-Lands": pygame.image.load(r"D:\Thi\Padm\Images\rewards\Kingdoms\River-lands.png"),
                "Kings-Landing": pygame.image.load(r"D:\Thi\Padm\Images\rewards\Kingdoms\Kings-landing.png"),
                "storm-lands": pygame.image.load(r"D:\Thi\Padm\Images\rewards\Kingdoms\Storm-lands.jpeg"),
            }
        }

        # Resize all images to tile size
        for items in ["Dragon", "Night-walker", "Army", "Daenerys-Targaryen", "Iron-throne", "Dragon-eggs"]:
            self.images[items] = pygame.transform.scale(self.images[items], (tile_size, tile_size))
        for k in self.kingdoms:
            self.images["kingdoms"][k] = pygame.transform.scale(self.images["kingdoms"][k], (tile_size, tile_size))

    def reset(self):
        self.agent_state = np.array([1, 1])
        return self.agent_state

    def step(self, action):
        proposed_state = self.agent_state.copy()

        if action == 0 and proposed_state[1] < self.grid_size - 1:  # up
            proposed_state[1] += 1
        elif action == 1 and proposed_state[1] > 0:  # down
            proposed_state[1] -= 1
        elif action == 2 and proposed_state[0] > 0:  # left
            proposed_state[0] -= 1
        elif action == 3 and proposed_state[0] < self.grid_size - 1:  # right
            proposed_state[0] += 1

        all_obstacles = sum(self.obstacles.values(), [])

        reward = 0
        done = np.array_equal(self.agent_state, self.goal_state)
        if done:
            reward = 100

        for egg_pos in self.rewards["Dragon-eggs"]:
            if np.array_equal(self.agent_state, egg_pos):
                reward += 10

        for k_pos in self.rewards["Kingdoms"]:
            if np.array_equal(self.agent_state, k_pos):
                reward += 20

        in_obstacle = any(np.array_equal(self.agent_state, obs) for obs in all_obstacles)
        if in_obstacle:
            reward = -10

        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        info = {"Distance to Goal": distance_to_goal}

        return self.agent_state, reward, done, info

    def render(self):
        self.screen.fill((255, 255, 255))  # white background

        for x in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (0, 0, 0), (x * self.tile_size, 0), (x * self.tile_size, self.screen_size))
        for y in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y * self.tile_size), (self.screen_size, y * self.tile_size))

        # Obstacles
        for group, coords in self.obstacles.items():
            for x, y in coords:
                self.screen.blit(self.images[group], (x * self.tile_size, y * self.tile_size))

        # Rewards
        for x, y in self.rewards["Dragon-eggs"]:
            self.screen.blit(self.images["Dragon-eggs"], (x * self.tile_size, y * self.tile_size))

        for name, (x, y) in zip(self.kingdoms, self.rewards["Kingdoms"]):
            self.screen.blit(self.images["kingdoms"][name], (x * self.tile_size, y * self.tile_size))

        # Agent
        ax, ay = self.agent_state
        self.screen.blit(self.images["Daenerys-Targaryen"], (ax * self.tile_size, ay * self.tile_size))

        # Goal
        gx, gy = self.goal_state
        self.screen.blit(self.images["Iron-throne"], (gx * self.tile_size, gy * self.tile_size))

        pygame.display.flip()
        pygame.time.wait(300)

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = PadmEnv(grid_size=10)
    state = env.reset()
    for _ in range(1000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                exit()

        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        if done:
            print("Goal reached!")
            break
    env.close()
