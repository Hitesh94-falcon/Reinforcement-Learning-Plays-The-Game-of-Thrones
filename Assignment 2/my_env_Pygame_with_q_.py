import gymnasium as gym
import numpy as np
import pygame


class PadmEnv(gym.Env):
    def __init__(self, grid_size=9, tile_size=64, render_mode=False):
        super().__init__()
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.render_mode = render_mode
        self.agent_state = np.array([1, 4])
        self.goal_state = np.array([7, 5])
        self.captured_kingdoms = set()

        self.action_space = gym.spaces.Discrete(4)

        self.obstacles = {
            "Dragon": [(1, 0), (8, 0), (1, 8),(0,4)],
            "Night-walker": [(0, 1), (4, 8)],
            "Army": [(8, 8), (3, 4), (5,2)]
        }

        self.rewards = {
            "Dragon-eggs": [(5, 0), (8, 6), (5, 7)],
            "Kingdoms": [(0,6), (2, 6), (4, 3), (7, 2),(2,2)]
        }


        self.kingdoms = ["Harrenhall", "River-Lands", "Kings-Landing", "storm-lands", "Westeros"]
        self.observation_space = gym.spaces.Box(low=0, high=grid_size, shape=(2,), dtype=np.int32)

        # Initialize Pygame
        if self.render_mode:
            pygame.init()
            self.screen_size = self.grid_size * self.tile_size
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("RL Plays Game of Thrones")

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
                    "Westeros": pygame.image.load(r"D:\Thi\Padm\Images\rewards\Kingdoms\Westeros.png")
                }
            }

            # Resize all images to tile size
            for items in ["Dragon", "Night-walker", "Army", "Daenerys-Targaryen", "Iron-throne", "Dragon-eggs"]:
                self.images[items] = pygame.transform.scale(self.images[items], (tile_size, tile_size))
            for k in self.kingdoms:
                self.images["kingdoms"][k] = pygame.transform.scale(self.images["kingdoms"][k], (tile_size, tile_size))
        else:
            self.images = {}

    def reset(self,random_initialization=True):
        if random_initialization:
            print("the agent is being intalized randomly")

            while True:
                x_rand = np.random.randint(0, self.grid_size)
                y_rand = np.random.randint(0, self.grid_size)
                if (x_rand, y_rand) not in sum(self.obstacles.values(), []):
                    self.agent_state = np.array([x_rand, y_rand])
                    break
        else:
            self.agent_state = np.array([1,4])
            
        self.captured_kingdoms.clear()
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

        self.agent_state = proposed_state.copy()

        all_obstacles = sum(self.obstacles.values(), [])

        reward = 0

        reached_goal = np.array_equal(self.agent_state, self.goal_state)
        all_kingdoms_captured = len(self.captured_kingdoms) == len(self.kingdoms)

        done = reached_goal and all_kingdoms_captured

        if reached_goal:
            if all_kingdoms_captured:
                reward = 100
            if len(self.kingdoms) == len(self.captured_kingdoms):
                print("All the kigdoms captured..... Valhalla")
            else:
                reward = -5  
                print("You must capture all kingdoms before reaching the throne!")


        for egg in self.rewards["Dragon-eggs"]:
            if np.array_equal(self.agent_state, egg):
                reward += 10

        for idx, k_pos in enumerate(self.rewards["Kingdoms"]):
            if np.array_equal(self.agent_state, k_pos):
                kingdom_name = self.kingdoms[idx]
                if kingdom_name not in self.captured_kingdoms:
                    self.captured_kingdoms.add(kingdom_name)
                    reward += 20    
        

        in_obstacle = any(np.array_equal(self.agent_state, obs) for obs in all_obstacles)
        if in_obstacle:
            reward = -15

        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        info = {"Distance to Goal": distance_to_goal}

        return self.agent_state, reward, done, info

    def render(self):
        if self.render_mode:
            print("in render....")
            self.screen.fill((255, 255, 255))  

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
            pygame.time.wait(30)

    def close(self):
        pygame.quit()

def create_env(grid_size=9, render_mode=False):
    return PadmEnv(grid_size=grid_size, render_mode=render_mode)
