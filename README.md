# RL Smart Taxi

Creating a smart taxi using reinforcement learning (Q-learning) to consistently and efficiently solve the taxi environment problem.

## The Taxi Environment

The "Taxi-v3" environment is part of the python Gymnasium library (forked from OpenAI's Gym library). 

<img src="images/taxi_env.png" width="700" />

There are four designated locations in the grid world indicated by Red, Green, Yellow, and Blue. When the episode starts, the taxi, the passenger, and the passenger's destination (shown by the building) start at different and randomized locations. The taxi must drive to the passenger’s initial location, pick up the passenger, then drive to the passenger’s destination, and then drop off the passenger. Once the passenger is dropped off, the episode ends. 
