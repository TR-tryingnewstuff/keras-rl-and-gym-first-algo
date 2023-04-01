# keras-rl-and-gym-first-algo
My first reinforcement learning algorithm using keras-rl and gym.

The algorithm is meant to train to trade using the S&P500 historical price data. 
Also pip install does not work with tvDatafeed, please refer to https://github.com/StreamAlpha/tvdatafeed  for more information or simply use the following command:
	
	pip install --upgrade --no-cache-dir git+https://github.com/StreamAlpha/tvdatafeed.git

There are 3 possible different actions it must choose from at every step : sell (0), do nothing (1), buy (2)

Rewards are based on profit / loss. If the action 1 was picked and there is low volatility, then it will receive a reward (to account for lower transaction costs), otherwise it will be penalized (missed opportunity).

Inputs : the observation includes the last 10 trading days' candles bodies, up wicks, down wicks and daily moves with 1% being equal to 1. 

Sequential model : The neural network was built using two LSTM layers first, flattening the outputs and then feeding the flattened tensor to a dense layer for choosing the action. The choice of the LSTM layer was based on the layer being able to better analyse sequences of data and find non-linear relationships compared to dense layers.


The script is rather simple and will be a good template for future models. 

Ideas to improve the script :

- Use a continuous action space to allow for position adjustments 
- Input more data in the observation (Open, high, low, close, macroeconomic indicators, daily/monthly dummies...) 
- Try different Neural network models (change number of layers, neurons, activation functions...) 
- Try a different agent 
