import time,os
import numpy as np
from my_env import create_env

q_table_dir = "q_tables"

q_table_kingdom1 = np.load(os.path.join(q_table_dir,"q_table_kingdom1.npy"))
q_table_kingdom2 = np.load(q_table_dir,"q_table_kingdom2.npy")
q_table_kingdom3 = np.load(q_table_dir,"q_table_kingdom3.npy")
q_table_goal     = np.load(q_table_dir,"q_table_goal.npy")


env = create_env(render_=True)


state = env.reset()
state = tuple(state)


trajectory = [state]
total_reward = 0
steps = 0
done = False
captured_kingdom1 = False
captured_kingdom2 = False
captured_kingdom3 = False


while not done:
    env.render()
    time.sleep(0.3)
    steps += 1


    if not captured_kingdom1:
        action = np.argmax(q_table_kingdom1[state])
    elif not captured_kingdom2:
        action = np.argmax(q_table_kingdom2[state])
    elif not captured_kingdom3:
        action = np.argmax(q_table_kingdom3[state])
    else:
        action = np.argmax(q_table_goal[state])

  
    next_state, reward, done, info = env.step(action)
    next_state = tuple(next_state)
    total_reward += reward
    trajectory.append(next_state)


    if info.get("Harrenhall captured") and not captured_kingdom1:
        captured_kingdom1 = True
        print("Harrenhall captured")

    if info.get("Riverlands captured") and not captured_kingdom2:
        captured_kingdom2 = True
        print("Riverlands captured")

    if info.get("KingsLanding captured") and not captured_kingdom3:
        captured_kingdom3 = True
        print("KingsLanding captured")

    if captured_kingdom1 and captured_kingdom2 and captured_kingdom3:
        print("Switching to GOAL Q-table")

    state = next_state


env.close()
print("Simulation finished")
print(f"Total reward: {total_reward}")
print(f"Steps taken: {steps}")
