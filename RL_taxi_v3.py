'''
Implementing RL for a Smartcab in the Gym Taxi-V3 Environment. -->
There are 4 locations (labeled by different letters), and our job is to pick up the passenger at one 
location and drop him off at another. We receive +20 points for a successful drop-off and lose 1 point 
for every time-step it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.
https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ --> Taxi v2 deprecated
'''

import gymnasium as gym
from gymnasium.wrappers import TimeLimit    #to prevent episode ending at 200 steps

from utils import solve_taxi_problem, train_q_learning

## Create and initialize environment
env = gym.make("Taxi-v3", render_mode="rgb_array")           #text-based rendering in terminal
env = TimeLimit(env.unwrapped, max_episode_steps=3000)       #prevent timeout at 200 steps
print("\nTaxi v3 environment initialized.\n")

## Randomly initialize env, print State and Action Space.
initial_state, _ = env.reset()                               #Need to reset first
env.unwrapped.s = initial_state                              #save initial state to use for both random and q-learning solutions

## Reward Table P. {action: [(probability, nextstate, reward, done)]}
# P = env.unwrapped.P[initial_state]                         #you can print the reward table for any state


'''
Solve using random action [infinite loop until end of epsiode (= drop-off)]:
'''
print("Solving with random actions ...")
epochs, penalties = solve_taxi_problem(env, initial_state, q_table=None, vidname="random_taxi.mp4")
print(f"Timesteps taken: {epochs}")
print(f"Penalties incurred: {penalties}\n")


'''
Training the agent --> Implementing Q-Learning to solve with RL: 
'''
q_table = train_q_learning(env)


'''
Solving Once with Q-learning: 
'''
print("Solving after q-learning ...")
epochs, penalties = solve_taxi_problem(env, initial_state, q_table=q_table, vidname="smart_taxi.mp4")
print("Timesteps taken: {}".format(epochs))
print("Penalties incurred: {}\n".format(penalties))


'''
Evaluate the Agent's performance after Q-learning: 
'''
total_epochs, total_penalties = 0, 0   
episodes = 100                          #evaluate on # episodes

print("Evaluating performance after Q-learning")
for _ in range(episodes):
    state, _ = env.reset()              #randomizes initial state for each episode 

    epochs, penalties = solve_taxi_problem(env, state, q_table=q_table, save_vid=False)

    total_penalties += penalties
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}\n")
