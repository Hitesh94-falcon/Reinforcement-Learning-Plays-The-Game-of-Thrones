import gymnasium as gym
import numpy as np
import pygame
import os


class PadmEnv(gym.Env):
    def __init__(self, grid_size=6, tile_size=64, render_ = False):
        super().__init__()
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.agent_state = np.array([0,0])
        self.goal_state = np.array([5,5])
        self.render_ = render_
        self.captured_kingdom1 = False
        self.captured_kingdom2 =False
        self.captured_kingdom3 = False
        self.goal_reached = False
        self.action_space = gym.spaces.Discrete(4)
        self.obstacles = [np.array([1,2]), np.array([3,0]), np.array([4,2]), np.array([3,5])]
        self.rewards = [np.array([0, 3]), np.array([5,1])]
        self.collected_egg = self.collected_eggs = {tuple(egg): False for egg in self.rewards}
        self.kingdom = [np.array([3,3]),np.array([1,5]),np.array([5,0])]

        self.observation_space = gym.spaces.Box(low=0, high=grid_size, shape=(2,), dtype=np.int32)

        # Initialize Pygame
        if self.render_:
            pygame.init()
            self.screen_size = self.grid_size * self.tile_size
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption("RL Plays Game of Thrones")

            # Load images
            image_dir = "Images"

            self.images = {
                "Dragon": pygame.image.load(os.path.join(image_dir, "obstacles", "dragon_balerion.png")),
                "Night-walker": pygame.image.load(os.path.join(image_dir, "obstacles", "nightwalker.png")),
                "Army": pygame.image.load(os.path.join(image_dir, "obstacles", "Army.png")),
                "Daenerys-Targaryen": pygame.image.load(os.path.join(image_dir, "states", "Rahnerya.png")),
                "Iron-throne": pygame.image.load(os.path.join(image_dir, "states", "iron_throne.png")),
                "Dragon-eggs": pygame.image.load(os.path.join(image_dir, "rewards", "Dragon-eggs.png")),
                "Harrenhall": pygame.image.load(os.path.join(image_dir, "rewards", "Kingdoms", "Harrenhall.jpeg")),
                "Riverlands": pygame.image.load(os.path.join(image_dir, "rewards", "Kingdoms", "River-lands.png")),
                "KingsLanding": pygame.image.load(os.path.join(image_dir, "rewards", "Kingdoms", "Kings-landing.png"))
            }

            # Resize all images to tile size
            for items in ["Dragon","Daenerys-Targaryen", "Iron-throne", "Dragon-eggs", "Night-walker", "Army","Harrenhall","Riverlands","KingsLanding"]:
                self.images[items] = pygame.transform.scale(self.images[items], (tile_size, tile_size))


    def reset(self,random_initialization = False):
        if random_initialization:
            print("the agent is being intalized randomly")

            while True:
                x_rand = np.random.randint(0, self.grid_size)
                y_rand = np.random.randint(0, self.grid_size)
                pos = np.array([x_rand, y_rand])
                if not (
                        any(np.array_equal(pos, obs) for obs in self.obstacles) or
                        np.array_equal(pos, self.goal_state) or
                        np.array_equal(pos, self.kingdom) or 
                        any(np.array_equal(pos, egg) for egg in self.rewards)):
                    self.agent_state = pos
                    break
        else:
            self.agent_state = np.array([0,0])
        self.captured_kingdom1 = False
        self.captured_kingdom2 =False
        self.goal_reached = False
        self.captured_kingdom3 =False
        self.collected_eggs = {tuple(egg): False for egg in self.rewards}
        return self.agent_state

    def step(self, action):
       
        if action == 0 and self.agent_state[1] < self.grid_size -1:  # up
                self.agent_state[1] += 1
        elif action == 1 and self.agent_state[1] > 0:  # down
            self.agent_state[1] -= 1
        elif action == 2 and self.agent_state[0] > 0:  # left
            self.agent_state[0] -= 1
        elif action == 3 and self.agent_state[0] < self.grid_size -1:# right
            self.agent_state[0] += 1


        reward = 0
        done = False
        
        if np.array_equal(self.agent_state, self.goal_state):
            if self.captured_kingdom1 and self.captured_kingdom2 and self.captured_kingdom3:
                reward += 100
                done = True
            else:
                reward = -2  
                print("You must capture all kingdoms before reaching the throne!")

        for egg in self.rewards:
            if np.array_equal(self.agent_state, egg):
                if not self.collected_eggs[tuple(egg)]:
                    reward += 5
                    self.collected_eggs[tuple(egg)] = True
                else:
                    reward -= 1  

        for kingdom_name, pos in zip(["Harrenhall", "Riverlands", "KingsLanding"], self.kingdom):
            if np.array_equal(self.agent_state, pos):
                if kingdom_name == "Harrenhall":
                    if not self.captured_kingdom1:
                        reward += 40
                        self.captured_kingdom1 = True
                    else:
                        reward -= 1
                elif kingdom_name == "Riverlands":
                    if not self.captured_kingdom2:
                        reward += 20
                        self.captured_kingdom2 = True
                    else:
                        reward -= 1
                elif kingdom_name == "KingsLanding":
                    if not self.captured_kingdom3:
                        reward += 20
                        self.captured_kingdom3 = True
                    else:
                        reward -= 1

        if True in [np.array_equal(self.agent_state, obs) for obs in self.obstacles]:
            reward = -50
            done = True

        reward += -0.1

        # Calculate distance to the goal for the info
        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        if self.captured_kingdom1 and self.captured_kingdom2 and self.captured_kingdom3:
            reward += 1.0 / (distance_to_goal + 1e-5)

        info = {"Distance to Goal": distance_to_goal,
                "Harrenhall captured": self.captured_kingdom1,
                 "Riverlands captured": self.captured_kingdom2,
                 "KingsLanding captured": self.captured_kingdom3, 
                 "Goal Reached": self.goal_reached}

        return self.agent_state, reward, done, info

    def render(self,render = None):
        if self.render_:
            self.screen.fill((255, 255, 255))  

            for x in range(self.grid_size + 1):
                pygame.draw.line(self.screen, (0, 0, 0), (x * self.tile_size, 0), (x * self.tile_size, self.screen_size))
            for y in range(self.grid_size + 1):
                pygame.draw.line(self.screen, (0, 0, 0), (0, y * self.tile_size), (self.screen_size, y * self.tile_size))

            # Obstacles
            for obs in self.obstacles:
                x, y = obs
                self.screen.blit(self.images["Dragon"], (x * self.tile_size, y * self.tile_size))

            # Rewards
            for x, y in self.rewards:
                self.screen.blit(self.images["Dragon-eggs"], (x * self.tile_size, y * self.tile_size))

            for pos, name in zip(self.kingdom, ["Harrenhall", "Riverlands", "KingsLanding"]):
                x, y = pos
                self.screen.blit(self.images[name], (x * self.tile_size, y * self.tile_size))

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


def create_env(render_= False):
    return PadmEnv(grid_size=6, render_=render_)

