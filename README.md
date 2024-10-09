# ACIT4610-24H-G13
ACIT4610-G13

# The project contain many tasks.

# Task 1
 Is to apply a Multi-Objective Evolutionary Algorithm (MOEA) to optimize traffic management strategies for selected New York City (NYC) areas. The goal is to minimize conflicting objectives, Total Travel Time (TTT) and Fuel Consumption (FC), using real-world traffic data from NYC Open Data.

# Data source: 
NYC Open Data

https://data.cityofnewyork.us/Transportation/Traffic-Volume-Counts/btm5-ppia/data_preview
python libraries: gym numpy tensorflow keras openAi Matplotlib requests pandas gym deap

# 



# Task 2
This Python code snippet uses the yfinance library to download historical stock price data and compute the monthly returns for a list of specified stock tickers. It begins by importing the necessary libraries: yfinance for financial data retrieval and pandas for data manipulation. A list of stock tickers is defined, representing major technology and consumer companies. The code also establishes a date range for downloading the historical stock data, spanning from January 1, 2020, to January 1, 2023. This setup prepares the environment to analyze the stock performance over a defined period.

The main functionality is encapsulated in a loop that iterates over each stock ticker in the list. Within the loop, the yf.download() function fetches the historical daily stock prices from Yahoo Finance. The daily stock prices are then resampled to calculate monthly returns, which are computed by using the percentage change of the adjusted closing prices. The monthly returns are stored in a new DataFrame, which is constructed iteratively. If the monthly_returns_combined DataFrame is empty, it initializes it with the returns of the first ticker. For subsequent tickers, it joins their monthly returns to the existing DataFrame, ensuring that all stock returns are consolidated into a single DataFrame for further analysis.

After calculating the monthly returns for all tickers, the code drops any rows containing missing values that might have resulted from stocks that did not have data for certain months. It then computes the covariance matrix of the monthly returns using the cov() method from pandas, which reveals the relationship between the returns of the different stocks. Finally, the covariance matrix is printed to the console, providing insights into how the stock returns move in relation to each other, which is particularly useful for portfolio optimization and risk assessment in financial analyses.
# Task 3
 Is to 

# Task 4
Taxi-v3 RL DQN, suitable small discrete state/action space.

Taxi location, passenger location, destination. Episode ends on passenger droped-off or time_limit.
Length of each episode is 200. Actions: pick-up, drop-off; movement north, south, east and west.

Transition probability, p, action_mask. Reward on correct drop-off. Penalty on delays and wrong actions.
-1 per step unless a reward.
-10 on bad “pickup” or “drop-off”.
+20 delivering the passenger.


