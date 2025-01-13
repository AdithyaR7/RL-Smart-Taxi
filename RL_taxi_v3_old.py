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

# Create and initialize environment
env = gym.make("Taxi-v3", render_mode="ansi")           #text-based rendering in terminal
env = TimeLimit(env.unwrapped, max_episode_steps=3000)


## Randomly initialize env, print State and Action Space.
# env.reset()     #Resets the environment and returns a random initial state.
# print(env.render())
# print("Action Space {}".format(env.action_space))
# print("State Space {}".format(env.observation_space))


## Render a specific state --> Used for testing 
state = env.unwrapped.encode(3, 1, 2, 0)  #(taxi row, taxi column, passenger index, destination index)
print("State:", state)                    #encode(3, 1, 2, 0) is state 328
env.reset()                               #Need to reset first
env.unwrapped.s = state
# print(env.render())                     #print env to terminal


## Reward Table P. {action: [(probability, nextstate, reward, done)]}
# P = env.unwrapped.P[328] 
# print(P)


'''Solve using random action (infinite loop until end of epsiode = drop-off)'''
env.unwrapped.s = 328   #set environment to illustration's state

epochs = 0
penalties, rewards = 0, 0   #initialize

frames = []             #for animation
done = False

while not done:         #infinite loop until solved
    action = env.action_space.sample()      #sample() picks a random action
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    if reward == -10:       #illegal pickup/drop-off
        penalties += 1
    
    # Put each rendered frame into dict for animation
    frames.append({
        'frame': env.render(),
        'state': state,
        'action': action,
        'reward': reward
        })

    epochs += 1

print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}".format(penalties))



## Print frames to terminal for visualization
def print_frames(frames):
    for i, frame in enumerate(frames):
        os.system('clear')              #clear terminal output
        print(frame['frame'])
        print(f"Timestep: {i + 1}")
        print(f"State: {frame['state']}")
        print(f"Action: {frame['action']}")
        print(f"Reward: {frame['reward']}")
        time.sleep(0.1)


# print_frames(frames)    #call to print taxi env to terminals


'''Implementing Q-Learning to solve with RL - Training the agent'''
q_table = np.zeros([env.observation_space.n, env.action_space.n])   #initialize to 0s

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# Plotting metrics
all_epochs = []
all_penalties = []
train_start = time.time()

for i in range(1, 100001):

    state, _ = env.reset()
    epochs, penalties, rewards = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space randomly
        else:
            action = np.argmax(q_table[state])  # Exploit learned values from Q table
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_q_value = (1 - alpha)*old_value + alpha*(reward + gamma*next_max)     
        q_table[state, action] = new_q_value  #update table
        
        if reward == -10:
            penalties += 1
        
        state = next_state
        epochs += 1
    
    if i % 100 == 0:
        os.system('clear')              #clear terminal output
        print(f"Episode: {i}")

train_end = time.time()
print("Training finished")
print(f"training time = {train_end - train_start}")


'''Evaluate the Agent's performance after Q-learning'''
total_epochs, total_penalties = 0, 0   
episodes = 100                          #evaluate on # episodes

for _ in range(episodes):
    state, _ = env.reset()
    epochs, penalties, rewards = 0, 0, 0
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")