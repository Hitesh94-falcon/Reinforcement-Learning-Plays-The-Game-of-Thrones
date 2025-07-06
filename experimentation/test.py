# import numpy as np
# from my_env import create_env
# import time

# # Load the trained Q-table
# q_table_kingdom = np.load("D:\Thi\Padm\Hail mary\q_value_folder\q_table_kingdom.npy")
# q_table_goal = np.load("D:\Thi\Padm\Hail mary\q_value_folder\q_table_goal.npy")

# # Create environment
# env = create_env()

# # Reset environment
# state = env.reset()
# state = tuple(state)

# # Simulation control variables
# done = False
# captured_kingdom = False

# # Run the simulation
# while not done:
#     env.render()
#     time.sleep(0.3)  # control speed of simulation

#     if not captured_kingdom:
#         action = np.argmax(q_table_kingdom[state])
#     else:
#         action = np.argmax(q_table_goal[state])

#     next_state, reward, done, _ = env.step(action)
#     next_state = tuple(next_state)

#     if env.captured_kingdom == True and not captured_kingdom:
#         captured_kingdom = True
#         print("kingdom captured Switching to goal Q-table")

#     state = next_state

# env.close()
# print("✅ Simulation finished")


import numpy as np
from my_env import create_env
import time

# Load Q-tables
q_table_kingdom = np.load(r"D:\Thi\Padm\experimentation\q_tables\q_table_kingdom.npy")
q_table_goal = np.load(r"D:\Thi\Padm\experimentation\q_tables\q_table_goal.npy")

# Create the environment
env = create_env()  # This will use render_=True as per your function



# Reset environment
state = env.reset()  # or True if desired
state = tuple(state)

trajectory = [state]
total_reward = 0
steps = 0
done = False
harrenhall_captured = False

while not done:
    env.render()
    time.sleep(0.5)
    steps += 1

    # Choose action greedily from correct Q-table
    if not harrenhall_captured:
        action = np.argmax(q_table_kingdom[state])
    else:
        action = np.argmax(q_table_goal[state])

    # Step the environment
    next_state, reward, done, info = env.step(action)
    next_state = tuple(next_state)
    total_reward += reward

    # Update kingdom flag
    if info.get("Harrenhall captured"):
        harrenhall_captured = True


    state = next_state

# Close and print results
env.close()
