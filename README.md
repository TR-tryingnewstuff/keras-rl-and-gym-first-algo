# keras-rl-and-gym-first-algo
My first reinforcement learning algorithm using keras-rl and gym.

The algorithm is meant to train to trade using the S&P500 historical price data. 
Also pip install does not work with tvDatafeed, please refer to https://github.com/StreamAlpha/tvdatafeed  for more information or simply use the following command:
	
	pip install --upgrade --no-cache-dir git+https://github.com/StreamAlpha/tvdatafeed.git

There are 3 possible different actions the agent must choose from at every step : sell (0), do nothing (1), buy (2)

Rewards are based on profit / loss.

Inputs : the observation includes the last 20 trading days' open, high, low, close, candle body (1% = 1) and relative range position (returns a value indicating current close position compared to the last 20 days lowest low and highest high with 0 meaning close == (highest high + lowest low) / 2 and -1 meaning close == lowest low).

Sequential model : The neural network was built using two LSTM layers first, flattening the outputs and then feeding the flattened tensor to a dense layer for choosing the action. The choice of the LSTM layer was based on the layer being able to better analyse sequences of data and find non-linear relationships compared to dense layers.


The script is rather simple and will be a good template for future models. 

Ideas to improve the script :

- Use a continuous action space to allow for position adjustments 
- Input more data in the observation (macroeconomic indicators, daily/monthly dummies...) 
- Try different Neural network models (change number of layers, neurons, activation functions...) 
- Try a different agent 
- Add mutliple indices / stocks over which the bot can act to allow for relative strength strategies, hedging and others
