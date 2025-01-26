# RL Smart Taxi

Creating a smart taxi using reinforcement learning (Q-learning) to consistently and efficiently solve the taxi environment problem.


## The Taxi Environment

The "Taxi-v3" environment is part of the python Gymnasium library (forked from OpenAI's Gym library). 

<img src="images/taxi_env.png" width="600" />

There are four designated locations in the grid world indicated by Red, Green, Yellow, and Blue. When the episode starts, the taxi, the passenger, and the passenger's destination (shown by the building) start at different and randomized locations. The taxi must drive to the passenger’s initial location, pick up the passenger, then drive to the passenger’s destination, and then drop off the passenger. Once the passenger is dropped off, the episode ends. 

In any state, there are 6 possible actions for the taxi (agent): North, East, South, West, Pickup, Dropoff. During an episode, the agent receives a 20 point reward for a successful drop-off and loses 1 point for every time-step it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.


## Random Solution vs Q-Learning

**Random Taxi** - here is the taxi environment being solved  by picking a random action at every state. This can take anywhere up to 3000+ epochs (after which the program is automatically stopped). This took a total of 681 epochs with 224 penalties incurred:

<img src="https://github.com/AdithyaR7/RL-Smart-Taxi/blob/main/taxi_sol_vids/random_taxi.gif" width="600" />

**Smart Taxi** - here is the agent efficiently carrying out moves to solve the problem in the most optimum way after Q-learning. This was solved in only 11 epochs with 0 penalties incurred:

<img src="https://github.com/AdithyaR7/RL-Smart-Taxi/blob/main/taxi_sol_vids/smart_taxi.gif" width="600" />


# Installation and Running the Code

This program was made using Python 3.10.16 in a Conda environment:
'''bash
conda create --name RL_Taxi python=3.10
'''

Use pip and requirements.txt to install the required packages:
'''bash
pip install requirements.txt
'''

Run the code:
'''bash
python RL_Taxi_v3.py
'''

This prints the relevant information to the terminal during the different sections of the code, and also saves the solutions of the random and smart taxi to video as .mp4 files.


# Acknowledgements

