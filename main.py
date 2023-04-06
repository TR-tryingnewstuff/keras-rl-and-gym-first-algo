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

WINDOW = 20

# ---------------------------------- Collecting S&P500 futures contract historical data ---------------------------------------------------

plot_df = pd.DataFrame(TvDatafeed().get_hist('ES1!', 'CME_MINI', Interval.in_daily, 5000))
df = pd.DataFrame(TvDatafeed().get_hist('ES1!', 'CME_MINI', Interval.in_daily, 5000))

df['relative_range_position'] = (df['close'] - ((df['low'].rolling(WINDOW).min() + df['high'].rolling(WINDOW).max()) /2)) / ((df['high'].rolling(WINDOW).max() - df['low'].rolling(WINDOW).min()) /2)
df['body'] = (df['close'] - df['open'])/ df['open']  * 100

df = df.drop(['symbol', 'volume'], axis = 1).dropna()

print(df.head(20))
n_columns = len(df.columns)


# ----------------------------- Defining the Market environment with the help of the gym.Env class -----------------------------------

class Market(gym.Env):
    def __init__(self):
        
        # Initializing the attributes
        
        self.action_space = Discrete(3)
        self.commission = 0
       
        self.done = False 
        self.n_step = 0
        self.index = np.random.randint(1, 4000)
        
        self.observation = np.asarray(df.iloc[self.index])
    
    
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
        self.observation = np.asarray(df.iloc[self.index])
        
        # Get reward
        
        if (self.action == 1):
            reward = 0
        
        elif (self.action == 0):
            reward = self.observation[-1] * -1 - self.commission

        elif (self.action == 2):
            reward = self.observation[-1] - self.commission
        
        
        if self.n_step == 500: 
            self.done = True
            
        info = {}
              
        return self.observation, reward, self.done, info
    
    
    def render(self, mode):
        
        # Information printing and plotting to visualize the process, significantly slows down training so it is not meant for extensive training
        # Set visualize=False to disable it
        
        print(f'  action taken : {self.action}')
        i = self.index
        color_index = np.where(plot_df.open[i:i+WINDOW] < plot_df.close[i:i+WINDOW], 'gray', 'black')
        date_index = np.array(plot_df[i:i+WINDOW].index)
        
        bars = np.array(plot_df.close[i:i+WINDOW])-np.array(plot_df.open[i:i+WINDOW])
        wicks = np.array(plot_df.high[i:i+WINDOW])-np.array(plot_df.low[i:i+WINDOW])
        plt.bar(date_index, bars, width=0.7, bottom=plot_df.open[i:i+WINDOW], color=color_index)
        plt.bar(date_index, wicks, width=0.1, bottom=plot_df.low[i:i+WINDOW], color=color_index)
        
        if self.action == 2:
            plt.arrow(x=date_index[-1], y=plot_df.high[i+WINDOW-1]+7, dx=0, dy=-4, width=0.2, color='green')
            
        elif self.action == 0:
            plt.arrow(x=date_index[-1], y=plot_df.high[i+WINDOW-1]+7, dx=0, dy=-4, width=0.2, color='red')
            
        plt.pause(0.1)
        plt.clf()
     
        
env = Market()

# -------------------------- Building the neural network architecture with the keras sequential model ------------------------------------

tf.compat.v1.experimental.output_all_intermediates(True)

def build_model(actions):
    """ Builds a Neural Network using the keras sequential model, takes the number \n
    of different possible actions available as input and returns a Keras model"""
      
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.LSTM(input_shape=(WINDOW, n_columns), units=WINDOW*n_columns, return_sequences=True))
    model.add(tf.keras.layers.LSTM(units=WINDOW*n_columns))
    model.add(tf.keras.layers.Flatten()) 
    model.add(tf.keras.layers.Dense(actions, activation='linear'))

    return model


actions = 3

# -------------------------------------------- Building the agent ------------------------------------------------------------------------

def build_agent(model, actions):
    """Builds the agent that will interact with the environment"""

    policy = rl.policy.BoltzmannQPolicy()
    memory = rl.memory.SequentialMemory(limit=10000, window_length=WINDOW)
    dqn = rl.agents.DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=50)
   
    return dqn
    
dqn = build_agent(build_model(actions), actions)

# ---------------------------------------- Compile and train ------------------------------------------------------------------------------

dqn.compile(optimizer='RMSprop')
dqn.fit(env, nb_steps=1000, verbose=1, visualize=True)

