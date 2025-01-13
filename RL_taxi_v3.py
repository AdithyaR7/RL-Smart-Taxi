'''
Implementing RL for a Smartcab in the Gym Taxi-V3 Environment. -->
There are 4 locations (labeled by different letters), and our job is to pick up the passenger at one 
location and drop him off at another. We receive +20 points for a successful drop-off and lose 1 point 
for every time-step it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ --> v2 deprecated
'''

import gymnasium as gym
from gymnasium.wrappers import TimeLimit    #to prevent episode ending at 200 steps

# from IPython.display import clear_output
import os 
import time
import numpy as np
import random 
import cv2

## Create and initialize environment
env = gym.make("Taxi-v3", render_mode="rgb_array")           #text-based rendering in terminal
env = TimeLimit(env.unwrapped, max_episode_steps=3000)       #prevent timeout at 200 steps
print("\nTaxi v3 environment initialized.\n")


## Randomly initialize env, print State and Action Space.
initial_state, _ = env.reset()                               #Need to reset first
env.unwrapped.s = initial_state                              #save initial state to use for both random and q-learning solutions

## Reward Table P. {action: [(probability, nextstate, reward, done)]}
# P = env.unwrapped.P[328] 
# print(P)

############################################################################
def record_taxi_episode(frames, epochs, penalties, filename="taxi_video.avi", fps=2.0, save=True):
    # Convert frames to video
    if save==True:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))   #use higher fps for random actions
        
        for frame in frames:
            # OpenCV uses BGR instead of RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Video saved as {filename}")

###############################################################################

def solve_taxi_problem(env, state, q_table=None, vidname="taxi_video.avi", save_vid=True):
    env.reset()                               #Need to reset first
    env.unwrapped.s = state

    epochs, penalties, reward = 0, 0, 0   #initialize 

    frames_random = []             #for animation
    frame = env.render()           #get and save first frame
    frames_random.append(frame)

    done = False                   #record end of episode = successfull drop-off

    while not done:                #infinite loop until solved

        if q_table is not None:
            action = np.argmax(q_table[state])      #use learned actions from q-table (q-learning)
            fps = 2.0

        else:
            action = env.action_space.sample()      #sample() picks a random action
            fps = 60.0
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if reward == -10:       #illegal pickup/drop-off
            penalties += 1

        # Get frame
        frame = env.render()
        frames_random.append(frame)
            
        epochs += 1

    record_taxi_episode(frames_random, epochs, penalties, filename=vidname, fps=fps, save=save_vid)
    
    return epochs, penalties
######################################################################################


'''Solve using random action (infinite loop until end of epsiode = drop-off)'''
print("Solving with random actions ...")

# env.reset()                               #Need to reset first
# env.unwrapped.s = initial_state

# epochs = 0
# penalties, reward = 0, 0   #initialize 

# frames_random = []             #for animation
# done = False

# # Get the first frame
# frame = env.render()
# frames_random.append(frame)

# while not done:         #infinite loop until solved
#     action = env.action_space.sample()      #sample() picks a random action
#     next_state, reward, terminated, truncated, _ = env.step(action)
#     done = terminated or truncated
    
#     if reward == -10:       #illegal pickup/drop-off
#         penalties += 1

#     # Get frame
#     frame = env.render()
#     frames_random.append(frame)
        
#     epochs += 1

# record_taxi_episode(frames_random, epochs, penalties, filename="random_taxi.mp4", fps=60.0)

epochs, penalties = solve_taxi_problem(env, initial_state, q_table=None, vidname="random_taxi.mp4")
print(f"Timesteps taken: {epochs}")
print(f"Penalties incurred: {penalties}\n")
#########################################################################


'''Implementing Q-Learning to solve with RL --> Training the agent'''
q_table = np.zeros([env.observation_space.n, env.action_space.n])   #initialize to 0s

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Plotting metrics
all_epochs = []
all_penalties = []
train_start = time.time()               #to measure total training time

print("Q-learning training started ...")
for i in range(1, 100001):

    state, _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  #explore action space randomly
        else:
            action = np.argmax(q_table[state])  #exploit learned values from Q table
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_q_value = (1 - alpha)*old_value + alpha*(reward + gamma*next_max)     
        q_table[state, action] = new_q_value    #update table
        
        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1
    

train_end = time.time()                 #end training time
print("Training finished")
print(f"Training time = {train_end - train_start} s\n")
###########################################################################################



'''Solving Once with Q-learning'''
# env.reset()                               #Need to reset first
# env.unwrapped.s = initial_state

# epochs, penalties, reward = 0, 0, 0   #initialize 

# frames_q = []             #for animation
# done = False

# # Get the first frame
# frame = env.render()
# frames_q.append(frame)

# while not done:
#     action = np.argmax(q_table[state])
#     state, reward, terminated, truncated, _ = env.step(action)
#     done = terminated or truncated

#     if reward == -10:
#         penalties += 1

#     # Get frame
#     frame = env.render()
#     frames_q.append(frame)

#     epochs += 1

# record_taxi_episode(frames_q, epochs, penalties, filename="smart_taxi.mp4", fps=2.0)          #MAKES VIDEO

print("Solving after q-learning ...")
epochs, penalties = solve_taxi_problem(env, initial_state, q_table=q_table, vidname="smart_taxi.mp4")
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}\n".format(penalties))
###################################


'''Evaluate the Agent's performance after Q-learning'''
total_epochs, total_penalties = 0, 0   
episodes = 100                          #evaluate on # episodes

print("Evaluating performance after Q-learning")
for _ in range(episodes):
    state, _ = env.reset()              #randomizes initial state for each episode 

    # while not done:
    #     action = np.argmax(q_table[state])
    #     state, reward, terminated, truncated, _ = env.step(action)
    #     done = terminated or truncated

    #     if reward == -10:
    #         penalties += 1

    #     epochs += 1

    epochs, penalties = solve_taxi_problem(env, state, q_table=q_table, save_vid=False)

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}\n")
######################################################################################

