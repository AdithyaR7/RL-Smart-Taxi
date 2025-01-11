'''
Implementing RL for a Smartcab in the Gym Taxi-V3 Environment. -->
There are 4 locations (labeled by different letters), and our job is to pick up the passenger at one 
location and drop him off at another. We receive +20 points for a successful drop-off and lose 1 point 
for every time-step it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ --> v2 deprecated
'''

import gymnasium as gym
from gymnasium.wrappers import TimeLimit    #to prevent episode ending at 200 steps

from IPython.display import clear_output
from time import sleep


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
print(env.render())


## Reward Table P. {action: [(probability, nextstate, reward, done)]}
# P = env.unwrapped.P[328] 
# print(P)


## Solve using random action (infinite loop until end of epsiode = drop-off) 
env.unwrapped.s = 328   #set environment to illustration's state

epochs = 0
penalties, rewards = 0, 0   #initialize

frames = []             #for animation
done = False

while not done:         #infinite loop until solved
    action = env.action_space.sample()      #sample() picks a random action
    observation, reward, terminated, truncated, info = env.step(action)
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
