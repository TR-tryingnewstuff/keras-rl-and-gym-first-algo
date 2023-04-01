# keras-rl-and-gym-first-algo
My first reinforcement learning algorithm using keras-rl and gym.

The algo is meant to train to trade using historical S&P500 price data. 
Two actions it must take at every step : buy (1) or sell (0)
Rewards are based on profit / loss.
Inputs : the observation includes the last 10 trading days' returns using pandas pct_change() function

Sequential model : The neural network was built using two LSTM layers first, flattening the outputs and then feeding the flattened tensor to a dense layer for choosing the action. The choice of the LSTM layer was based on the layer being able to better analyse sequences of data and find non-linear relationships compared to dense layers.

The script is rather simple and will be a good template for future models. 

Ideas to improve the script :
    - Use a non discrete action space to allow for position adjustments and not doing anything \n
    - Input more data in the observation (Open, high, low, close, macroeconomic indicators, daily/monthly dummies...) \n
    - Try different Neural network models (change number of layers, neurons, activation functions...) \n
    - Try different agent \n
