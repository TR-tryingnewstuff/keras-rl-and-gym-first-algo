
import numpy as np
import pandas as pd

import gym
from gym.spaces import Discrete, Box
from tvDatafeed import TvDatafeed, Interval

import keras
import tensorflow as tf
import rl.agents
import rl.policy
import rl.memory
import matplotlib.pyplot as plt

# ---------------------------------- Collecting S&P500 futures contract historical data ---------------------------------------------------

df = pd.DataFrame(TvDatafeed().get_hist('ES1!', 'CME_MINI', Interval.in_daily, 5000)).close.pct_change() * 100
full_df = pd.DataFrame(TvDatafeed().get_hist('ES1!', 'CME_MINI', Interval.in_daily, 5000))


# ----------------------------- Defining the Market environment with the help of the gym.Env class -----------------------------------

class Market(gym.Env):
    def __init__(self, render_mode=None):
        
        # Initializing the attributes
        
        self.action_space = Discrete(2)
       
        self.done = False 
        self.n_step = 0
        self.index = np.random.randint(1, 2000)
        
        self.observation = np.asarray(df.iloc[self.index+10:self.index+20])
    
        self.observation_space = Box(low=-1, high=1, shape=(10,))
    
    def reset(self):
        
        # Resets the environment with a random start index to shuffle the data and avoid that the algo learns the sequence
        
        self.done = False
        self.n_step = 0
        self.index = np.random.randint(1, 4000)
        
        return self.observation
         
        
    def step(self, action):
         
        self.action = action
        self.index += 1
        self.n_step += 1
        self.observation = np.asarray(df.iloc[self.index+10:self.index+20])
        
        # Get reward
        
        if (self.action == 1) and (self.observation[-1] > 0):
            reward = self.observation[-1] 
        
        elif (self.action == 1) and (self.observation[-1] < 0):
            reward = self.observation[-1]  
            
        elif (self.action == 0) and (self.observation[-1] < 0):
            reward = -self.observation[-1]

        elif (self.action == 0) and (self.observation[-1] > 0):
            reward = -self.observation[-1]     

        else:
            reward = 0
        
        
        if self.n_step == 100: 
            self.done = True
            
        info = {}
        
        # Information printing and plotting to visualize the process, significantly slows down the process so it is not meant for extensive training
        
        print(f'  action taken : {self.action}')
        i = self.index
        color_index = np.where(full_df.open[i+10:i+20] < full_df.close[i+10:i+20], 'gray', 'black')
        date_index = np.array(full_df[i+10:i+20].index)
        
        bars = np.array(full_df.close[i+10:i+20])-np.array(full_df.open[i+10:i+20])
        wicks = np.array(full_df.high[i+10:i+20])-np.array(full_df.low[i+10:i+20])
        plt.bar(date_index, bars, width=0.7, bottom=full_df.open[i+10:i+20], color=color_index)
        plt.bar(date_index, wicks, width=0.1, bottom=full_df.low[i+10:i+20], color=color_index)
        
        if self.action == 1:
            plt.arrow(x=date_index[-1], y=full_df.high[i+19]+7, dx=0, dy=-4, width=0.2, color='green')
            
        else:
            plt.arrow(x=date_index[-1], y=full_df.high[i+19]+7, dx=0, dy=-4, width=0.2, color='red')
            
        plt.pause(0.1)
        plt.clf()
        
        return self.observation, reward, self.done, info
    
    def render(self, mode):
         # TO DO : add rendering 
         
        pass
        
     
        
env = Market()

# -------------------------- Building the neural network architecture with the keras sequential model ------------------------------------

tf.compat.v1.experimental.output_all_intermediates(True)

def build_model(actions):
    """ Builds a Neural Network using the keras sequential model, takes the number \n
    of different possible actions available as input and returns a Keras model"""
      
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.LSTM(input_shape=(10, 10), units=10, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=10, activation='sigmoid'))
    model.add(tf.keras.layers.Flatten()) 
    model.add(tf.keras.layers.Dense(actions, activation='linear'))

    return model


actions = 2

# -------------------------------------------- Building the agent ------------------------------------------------------------------------

def build_agent(model, actions):
    """Builds the agent that will interact with the environment"""

    policy = rl.policy.BoltzmannQPolicy()
    memory = rl.memory.SequentialMemory(limit=10000, window_length=10)
    dqn = rl.agents.DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=50)
    
    return dqn
    
dqn = build_agent(build_model(actions), actions)

# ---------------------------------------- Compile and train ------------------------------------------------------------------------------

dqn.compile(optimizer='RMSprop')
dqn.fit(env, nb_steps=1000, visualize=True, verbose=1)

