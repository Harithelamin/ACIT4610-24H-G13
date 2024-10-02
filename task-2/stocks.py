import yfinance as yf
import pandas as pd

# List of stock tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 'V', 'JNJ',
           'JPM', 'PG', 'UNH', 'HD', 'DIS', 'MA', 'PEP', 'KO', 'PFE', 'XOM']

# Define date range
start_date = '2020-01-01'  # example start date
end_date = '2023-01-01'    # example end date

# Create an empty DataFrame to store the monthly returns of all stocks
monthly_returns_combined = pd.DataFrame()

# Download historical daily stock prices from Yahoo Finance
for ticker in tickers:
    # Download stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # Resample daily data to month-end and calculate monthly returns
    stock_data['Monthly Return'] = stock_data['Adj Close'].resample('ME').ffill().pct_change()
    
    # Store monthly returns in a DataFrame
    if monthly_returns_combined.empty:
        monthly_returns_combined = stock_data[['Monthly Return']].rename(columns={'Monthly Return': ticker})
    else:
        monthly_returns_combined = monthly_returns_combined.join(stock_data[['Monthly Return']].rename(columns={'Monthly Return': ticker}))

# Drop rows with missing values (if any)
monthly_returns_combined.dropna(inplace=True)

# Calculate the covariance matrix of the monthly returns
covariance_matrix = monthly_returns_combined.cov()

# Print the covariance matrix
print("Covariance Matrix:")
print(covariance_matrix)
