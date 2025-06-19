import gymnasium as gym
import numpy as np
import pygame
import threading
import os

class PadmEnv(gym.Env):
    def __init__(self, grid_size=9, cell_size=70):
        super().__init__()

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_width = self.grid_size * self.cell_size 
        self.screen_height = self.grid_size * self.cell_size 

        self.agent_state = np.array([5,0])
        self.goal_state = np.array([4,6])
        self.action_space = gym.spaces.Discrete(4)

        self.obstacles = {
                    # Original (x, y) -> Converted (row, column)
                    "Dragon": [(0,1), (8,1), (0,8)], 
                    "Night-walker": [(8,4), (1,0)], 
                    "Army": [(4,3), (8,8), (2,5)] 
                }

        self.rewards = {
                    
                    "Dragon-eggs": [(7,5), (6,8), (0,5)], 
                    "Kingdoms": [(3,1), (6,2), (3,4), (2,7)]
                }

        self.kingdoms = ["Harrenhall", "River-Lands", "Kings-Landing", "storm-lands"]

        self.visited_kingdoms_coords = set()
        self._initial_kingdoms_coords = {tuple(cord) for cord in self.rewards["Kingdoms"]} 

        self.image_paths = {
            "Dragon": r"D:\Thi\Padm\Images\obstacles\dragon_balerion.png", # Changed from Army.png
            "Night-walker": r"D:\Thi\Padm\Images\obstacles\nightwalker.png",
            "Daenerys-Targaryen": r"D:\Thi\Padm\Images\states\Rahnerya.png",
            "Iron-throne": r"D:\Thi\Padm\Images\states\iron_throne.png",
            "Army": r"D:\Thi\Padm\Images\obstacles\Army.png",
            "Dragon-eggs": r"D:\Thi\Padm\Images\rewards\Dragon-eggs.png",
            # "background": r"Images/BG/Background.png",

            "kingdoms": {
                "Harrenhall": r"D:\Thi\Padm\Images\rewards\Kingdoms\Harrenhall.jpeg",
                "River-Lands": r"D:\Thi\Padm\Images\rewards\Kingdoms\River-lands.png",
                "Kings-Landing": r"D:\Thi\Padm\Images\rewards\Kingdoms\Kings-landing.png",
                "storm-lands": r"D:\Thi\Padm\Images\rewards\Kingdoms\Westeros.png"
            }
        }

        self.images = {}

        self.observation_space = gym.spaces.Box(
            low=0, high=self.grid_size, shape=(2,), dtype=np.int32
        )
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Game of Thrones")
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.Font(None, 36)

        self.image_loading_thread()
        

    def image_loading_thread(self):
        def load_images():
            try:
                
                self.images["kingdoms"] = {} 
                
                for key, path_value in self.image_paths.items(): 
                    if key == "kingdoms":
                        for key_name, k_path in path_value.items():
                            if not os.path.exists(k_path):
                                print(f"Warning: Kingdom image not found: {k_path}")
                                continue
                            img = pygame.image.load(k_path).convert_alpha()
                            self.images[key][key_name] = pygame.transform.scale(img, (self.cell_size, self.cell_size))
                    else:
                        if not os.path.exists(path_value):
                            print(f"Warning: Image not found: {path_value}")
                            continue
                        img = pygame.image.load(path_value).convert_alpha()
                        self.images[key] = pygame.transform.scale(img, (self.cell_size, self.cell_size))
                        
                        # if "background" in self.image_paths and os.path.exists(self.image_paths["background"]):
                        #     bg_img = pygame.image.load(self.image_paths["background"]).convert_alpha()
                        #     self.images["background"] = pygame.transform.scale(bg_img, (self.screen_width, self.screen_height))
                        #     self.images["background"].set_alpha(255)

                print("All images loaded successfully.")
            except pygame.error as e:
                print(f"Error loading image during _start_image_loading_thread: {e}")
                print("This might be due to incorrect paths or corrupted files.")
                pygame.quit()
                exit()

        thread = threading.Thread(target=load_images)
        thread.start()
        thread.join()

    def reset(self):
        self.agent_state = np.array([5,0])
        return self.agent_state

    def step(self, action):
        proposed_state = self.agent_state.copy()
        reward = 0
        done = False
        
        if action == 0:  # Right
            proposed_state[1] += 1
        elif action == 1:  # Left
            proposed_state[1] -= 1
        elif action == 2:  # Up
            proposed_state[0] -= 1
        elif action == 3:  # Down
            proposed_state[0] += 1

        if not (0 <= proposed_state[0] < self.grid_size and 
                0 <= proposed_state[1] < self.grid_size):
            
            distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
            info = {"Distance to Goal": distance_to_goal}
            return self.agent_state, reward, done, info

        # Obstacle Check: Check if the proposed state is an obstacle
        in_obstacle = False
        for obs_list in self.obstacles.values():
            if any(np.array_equal(proposed_state, obs_coord) for obs_coord in obs_list):
                in_obstacle = True
                break

        if in_obstacle:
            reward = -10 
            distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state, ord=1)
            info = {"Distance to Goal": distance_to_goal, "Hit Obstacle": True}
            return self.agent_state, reward, done, info

        self.agent_state = proposed_state.copy()

        for kingdoms_cord in self.rewards["Kingdoms"]:
            if np.array_equal(self.agent_state,kingdoms_cord):
                if tuple(kingdoms_cord) not in self.visited_kingdoms_coords:
                    self.visited_kingdoms_coords.add(tuple(kingdoms_cord))
                    reward += 50
                break
             
           
        # 3. Reward/Goal Check 
        if np.array_equal(self.agent_state, self.goal_state):
            if len(self.visited_kingdoms_coords) == len(self.rewards["Kingdoms"]):
                   reward = 100
                   done = True
                   print("All the kingdoms captured")
            else:
                reward -= 5
                self.reset()
                print("Goal is reached but all kingdoms are not captured resetting the agent...")

        else:
            # Check for other rewards 
            is_reward_location = False
            for reward_type, reward_coords_list in self.rewards.items():
                if any(np.array_equal(self.agent_state, rew_coord) for rew_coord in reward_coords_list):
                    is_reward_location = True
                    break
            
            if is_reward_location:
                reward = 20 
            else:
                reward = 0 

        # Calculate info for the current step
        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state, ord=1)
        info = {"Distance to Goal": distance_to_goal, "Hit Wall": False, "Hit Obstacle": False} # Reset flags

        return self.agent_state, reward, done, info

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
            
        if "background" in self.images:
            self.screen.blit(self.images["background"],(0,0))
        else:
            self.screen.fill((255, 255, 255))

        # Draw vertical grid lines
        for x in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (0, 0, 0),
            (x * self.cell_size, 0), 
            (x * self.cell_size, self.grid_size * self.cell_size),1) 
        
        # Draw horizontal grid lines
        for y in range(self.grid_size + 1):
            pygame.draw.line(self.screen, (0, 0, 0),
            (0, y * self.cell_size), 
            (self.grid_size * self.cell_size, y * self.cell_size),1) 

        # Draw obstacles
        for obstacle_type, coords_list in self.obstacles.items():
            if obstacle_type in self.images:
                for r, c in coords_list:
                    self.screen.blit(self.images[obstacle_type], (c * self.cell_size, r * self.cell_size))

        # Draw Dragon-eggs
        for r, c in self.rewards["Dragon-eggs"]:
            if "Dragon-eggs" in self.images:
                self.screen.blit(self.images["Dragon-eggs"], (c * self.cell_size, r * self.cell_size))

        # Draw Kingdoms
        for i, (r, c) in enumerate(self.rewards["Kingdoms"]):
            name = self.kingdoms[i]
            if "kingdoms" in self.images and name in self.images["kingdoms"]:
                self.screen.blit(self.images["kingdoms"][name], (c * self.cell_size, r * self.cell_size))

        # Draw Agent
        agent_r, agent_c = self.agent_state
        if "Daenerys-Targaryen" in self.images:
            self.screen.blit(self.images["Daenerys-Targaryen"], (agent_c * self.cell_size, agent_r * self.cell_size))

        # Draw Goal
        goal_r, goal_c = self.goal_state
        if "Iron-throne" in self.images:
            self.screen.blit(self.images["Iron-throne"], (goal_c * self.cell_size, goal_r * self.cell_size))

        pygame.display.flip()       
        self.clock.tick(5)

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = PadmEnv(grid_size=9, cell_size=60)
    state = env.reset()
    running = True
    for _ in range(1000):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        if not running:
            break

        try:
            env.render()
        except SystemExit:
            running = False
            break

        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")

        if done:
            print("Goal reached!")
            break

    env.close()