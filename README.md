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


# Installation

This program was made using Python 3.10.16 in a Conda environment:
```bash
conda create --name RL_Taxi python=3.10
```

Use pip and requirements.txt to install the required packages:
```bash
pip install -r requirements.txt

```

Errors - If you encounter the error message below:
```
libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: iris
libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
libGL error: failed to load driver: swrast
X Error of failed request:  BadValue (integer parameter out of range for operation)
  Major opcode of failed request:  149 (GLX)
  Minor opcode of failed request:  3 (X_GLXCreateContext)
  Value in failed request:  0x0
  Serial number of failed request:  102
  Current serial number in output stream:  103
```

then run the following to fix this:
```bash
conda install -c conda-forge libstdcxx-ng
```

# Running the Code

Run the python code:
```bash
python RL_Taxi_v3.py
```

This prints the relevant information to the terminal during the different sections of the code, and also saves the solutions of the random and smart taxi to video as .mp4 files.




# Acknowledgements

This code was largely adapted from an article from learndatasci.com and updated to use v3 of the Taxi environment instead of the now deprecated v2.

--> https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/ 

--> https://www.gymlibrary.dev/environments/toy_text/taxi/

--> https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris 
