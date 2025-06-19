import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg




class PadmEnv(gym.Env):
    def __init__(self, grid_size=10):
        super().__init__()
        self.grid_size = grid_size
        self.agent_state = np.array([1,4])
        self.goal_state = np.array([7,5])
        self.action_space = gym.spaces.Discrete(4)
       
        self.obstacles = {
            "Dragon": [(2,1),(2,9),(9,9)],
            "Night-walker":[(1,8),(5,1)],
            "Army":[(4,5),(6,7),(9,2)]
            
        }


        self.rewards = {
            "Dragon-eggs": [(6,2),(6,9),(9,4)],
            "Kingdoms": [(3,3),(2,6),(5,6),(8,7)]

        }

        self.kingdoms = ["Harrenhall","River-Lands","Kings-Landing","storm-lands"]
        
        self.images = {
            "Dragon": mpimg.imread(r"D:\Thi\Padm\Images\obstacles\dragon_balerion.png"),
            "Night-walker": mpimg.imread(r"D:\Thi\Padm\Images\obstacles\nightwalker.png"),
            "Daenerys-Targaryen": mpimg.imread(r"D:\Thi\Padm\Images\states\Rahnerya.png"),
            "Iron-throne": mpimg.imread(r"D:\Thi\Padm\Images\states\iron_throne.png"),
            "Army": mpimg.imread(r"D:\Thi\Padm\Images\obstacles\Army.png"),
            "Dragon-eggs": mpimg.imread(r"D:\Thi\Padm\Images\rewards\Dragon-eggs.png"),

            "kingdoms": {
            "Harrenhall": mpimg.imread(r"D:\Thi\Padm\Images\rewards\Kingdoms\Harrenhall.jpeg"),
            "River-Lands": mpimg.imread(r"D:\Thi\Padm\Images\rewards\Kingdoms\River-lands.png"),
            "Kings-Landing": mpimg.imread(r"D:\Thi\Padm\Images\rewards\Kingdoms\Kings-landing.png"),
            "storm-lands": mpimg.imread(r"D:\Thi\Padm\Images\rewards\Kingdoms\Storm-lands.jpeg"), 

            } 
            
        }

        self.observation_space = gym.spaces.Box(
            low=0, high=self.grid_size, shape=(2,), dtype=np.int32
        )
        self.fig, self.ax = plt.subplots(figsize=(15, 15))  # Increase from (6,6) or default
        plt.ion()
        plt.show()

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
        if not any(np.array_equal(proposed_state, obs) for obs in all_obstacles):
            self.agent_state = proposed_state

        reward = 0
        done = np.array_equal(self.agent_state, self.goal_state)
        if done:
            reward = 10

        for egg_pos in self.rewards["Dragon-eggs"]:
            if np.array_equal(self.agent_state, egg_pos):
                reward += 2

        for k_pos in self.rewards["Kingdoms"]:
            if np.array_equal(self.agent_state, k_pos):
                reward += 3

        in_obstacle = any(np.array_equal(self.agent_state, obs) for obs in all_obstacles)
        if in_obstacle:
            reward = -10

        distance_to_goal = np.linalg.norm(self.goal_state - self.agent_state)
        info = {"Distance to Goal": distance_to_goal}
        
        return self.agent_state, reward, done, info


    def render(self):
        self.ax.clear()

        # Draw grid
        for x in range(self.grid_size + 2):
            self.ax.axvline(x, color='black',linewidth=1)
        for y in range(self.grid_size + 2):
            self.ax.axhline(y, color='black',linewidth=1)

        # Draw-Images of obstacles
        for idx, (x, y) in enumerate(self.obstacles["Dragon"]):
            self.ax.imshow(self.images["Dragon"],extent=[x,x+1,y,y+1])

        for idx, (x, y) in enumerate(self.obstacles["Night-walker"]):
            self.ax.imshow(self.images["Night-walker"],extent=[x,x+1,y,y+1])
        
        for idx, (x,y) in enumerate(self.obstacles["Army"]):
            self.ax.imshow(self.images["Army"],extent=[x,x+1,y,y+1])
        
        for idx, (x,y) in enumerate(self.rewards["Dragon-eggs"]):
            self.ax.imshow(self.images["Dragon-eggs"],extent=[x,x+1,y,y+1])
        
        for names ,(x,y) in zip(self.kingdoms,self.rewards["Kingdoms"]):
                img = self.images["kingdoms"][names]
                self.ax.imshow(img, extent=(x, x+1,y,y+1))


        # Draw agent
        agent_x ,agent_y = self.agent_state
        self.ax.imshow(self.images["Daenerys-Targaryen"],extent=[agent_x,agent_x+1,agent_y,agent_y+1])

        # Draw goal
        goal_x, goal_y = self.goal_state
        self.ax.imshow(self.images["Iron-throne"],extent=[goal_x,goal_x+1,goal_y,goal_y+1])

        
        self.ax.set_xlim(0, self.grid_size) 
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.fig.canvas.draw()
        plt.pause(0.9)

if __name__ == "__main__":
    steps = 1000  
    env = PadmEnv(grid_size=10)
    state, _ = env.reset()
    env.render()
    
    for _ in range(steps):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}, Info: {info}")
        
        if done:
            print("Goal reached!")
            break
    env.close() 


