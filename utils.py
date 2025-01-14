'''Utils.py contains functions for solving the taxi probelm and saving episodes as a video for visualization.'''

import numpy as np
import cv2
import time
import random


def solve_taxi_problem(env, state, q_table=None, vidname="taxi_video.avi", save_vid=True):

    env.reset()                               #Need to reset first
    env.unwrapped.s = state

    epochs, penalties, reward = 0, 0, 0       #initialize 

    frame = env.render()                      #get first frame

    frame_info = {
        "frames":[frame],                               #frames for animation. Save first frame
        "epochs":[f"Epochs:  {epochs}"],                 #string for writing on video
        "penalties":[f"Penalties:{penalties}"]
    }

    done = False                              #record end of episode = successfull drop-off

    while not done:                           #infinite loop until solved

        if q_table is not None:
            action = np.argmax(q_table[state])          #use learned actions from q-table (q-learning)
            fps = 2.0

        else:
            action = env.action_space.sample()          #sample() picks a random action
            fps = 60.0
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        if reward == -10:                      #illegal pickup/drop-off
            penalties += 1

        frame = env.render()                             #get and save frame, epoch & penalty string
        frame_info["frames"].append(frame)
        frame_info["epochs"].append(f"Epochs:  {epochs}")
        frame_info["penalties"].append(f"Penalties:{penalties}")
            
        epochs += 1

    record_taxi_episode(frame_info, filename=vidname, fps=fps, save=save_vid)
    
    return epochs, penalties


def train_q_learning(env):

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
    print("Training complete")
    print(f"Training time = {train_end - train_start} s\n")

    return q_table


def record_taxi_episode(frame_info, filename="taxi_video.avi", fps=2.0, save=True):

    #extract lists from frame_info dict
    frames = frame_info["frames"]
    epochs = frame_info["epochs"]
    penalties = frame_info["penalties"]

    # Convert frames to video
    if save==True:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))   #use higher fps for random actions
        
        for frame, epoch_str, penalties_str in zip(frames, epochs, penalties):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)          #OpenCV uses BGR instead of RGB
        
            # Add overlay text to the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (255, 255, 255)
            thickness = 1

            cv2.putText(frame_bgr, epoch_str,     (101,55), font, font_scale, color, thickness)
            cv2.putText(frame_bgr, penalties_str, (101,72), font, font_scale, color, thickness)

            out.write(frame_bgr)                                        #write to frame

        out.release()
        print(f"Video saved as {filename}")
        
    return


def save_initial_state_png(env):

    # Render the current frame
    frame = env.render()
    
    # Convert the frame to BGR (required for OpenCV)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Save the frame as an image
    cv2.imwrite("initial_state.png", frame_bgr)

    return