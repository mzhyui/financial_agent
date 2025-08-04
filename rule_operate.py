import os
import pandas as pd
import numpy as np

def calculate_kdj(df, n=9, m1=3, m2=3):
    """Calculate KDJ indicator"""
    df = df.copy()
    
    # Calculate RSV (Raw Stochastic Value)
    low_min = df['low'].rolling(window=n).min()
    high_max = df['high'].rolling(window=n).max()
    df['rsv'] = (df['close'] - low_min) / (high_max - low_min) * 100
    
    # Calculate K, D, J values
    df['k'] = df['rsv'].ewm(alpha=1/m1).mean()
    df['d'] = df['k'].ewm(alpha=1/m2).mean()
    df['j'] = 3 * df['k'] - 2 * df['d']
    
    return df

def calculate_technical_indicators(df):
    """Calculate additional technical indicators for MCDA"""
    df = df.copy()
    
    # Price momentum
    df['price_change'] = df['close'].pct_change()
    df['price_momentum'] = df['close'].rolling(5).mean() / df['close'].rolling(20).mean() - 1
    
    # Volatility
    df['volatility'] = df['close'].rolling(10).std()
    
    # Moving averages
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma_signal'] = np.where(df['ma5'] > df['ma20'], 1, -1)
    
    return df

def mcda_scoring(df):
    """Multi-Criteria Decision Analysis scoring"""
    df = df.copy()
    
    # Normalize indicators to 0-1 scale
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min())
    
    # KDJ signals (weight: 40%)
    kdj_buy_signal = ((df['k'] > df['d']) & (df['j'] > 80)).astype(int)
    kdj_sell_signal = ((df['k'] < df['d']) & (df['j'] < 20)).astype(int)
    df['kdj_score'] = kdj_buy_signal - kdj_sell_signal
    
    # Price momentum (weight: 30%)
    df['momentum_score'] = np.where(df['price_momentum'] > 0.02, 1, 
                                   np.where(df['price_momentum'] < -0.02, -1, 0))
    
    # Moving average signal (weight: 20%)
    df['ma_score'] = df['ma_signal']
    
    # Volatility adjustment (weight: 10%)
    df['vol_score'] = np.where(df['volatility'] < df['volatility'].rolling(20).mean(), 1, -1)
    
    # MCDA weighted score
    weights = {'kdj': 0.4, 'momentum': 0.3, 'ma': 0.2, 'vol': 0.1}
    df['mcda_score'] = (weights['kdj'] * df['kdj_score'] + 
                        weights['momentum'] * df['momentum_score'] + 
                        weights['ma'] * df['ma_score'] + 
                        weights['vol'] * df['vol_score'])
    
    return df

def generate_trading_signals(df, mcda_threshold=0.3):
    """Generate trading decisions based on MCDA and KDJ"""
    df = df.copy()
    
    # Trading rules
    buy_condition = (
        (df['mcda_score'] > mcda_threshold) & 
        (df['k'] > df['d']) & 
        (df['j'] > 20) & (df['j'] < 80) &
        (df['close'] > df['ma5'])
    )
    
    sell_condition = (
        (df['mcda_score'] < -mcda_threshold) | 
        (df['j'] > 90) | 
        (df['k'] < df['d']) & (df['j'] < 20)
    )
    
    # Position sizing based on signal strength
    df['signal_strength'] = abs(df['mcda_score'])
    df['base_hands'] = np.where(df['signal_strength'] > 0.5, 2000, 1500)
    
    # Generate decisions
    df['Decision'] = 0
    df['Hands'] = 0
    
    position = 0
    for i in range(len(df)):
        if buy_condition.iloc[i] and position <= 0:
            df.iloc[i, df.columns.get_loc('Decision')] = 1
            df.iloc[i, df.columns.get_loc('Hands')] = df.iloc[i, df.columns.get_loc('base_hands')]
            position = 1
        elif sell_condition.iloc[i] and position >= 0:
            df.iloc[i, df.columns.get_loc('Decision')] = -1
            df.iloc[i, df.columns.get_loc('Hands')] = df.iloc[i, df.columns.get_loc('base_hands')]
            position = -1
        else:
            df.iloc[i, df.columns.get_loc('Decision')] = 0
            df.iloc[i, df.columns.get_loc('Hands')] = 0
    
    return df

# Sample usage with your data format
def process_trading_data(csv_data):
    """Process CSV data and generate trading signals"""
    
    # Read the data
    df = pd.read_csv(csv_data) if isinstance(csv_data, str) else csv_data
    
    # Calculate KDJ indicators
    df = calculate_kdj(df)
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Apply MCDA scoring
    df = mcda_scoring(df)
    
    # Generate trading signals
    df = generate_trading_signals(df)
    
    # Return results in requested format
    result = df[['Decision', 'Hands']].copy()
    
    # Convert Decision values: 1 for buy, -1 for sell, 0 for hold
    # Adjust to your format where 1=action, 0=no action
    # result['Decision'] = result['Decision'].apply(lambda x: 1 if x != 0 else 0)
    result['Hands'] = result['Hands'].astype(int)
    
    return result

    # Load CSV data

def getStockMarketData(ticker):
    '''
    timestamp,open,high,low,close,volume
    '''
    # If you have already saved data, just load it from the file
    df = pd.read_csv(f'./data/stock_market_data-{ticker}.csv', parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Clean the data - remove rows where OHLC values are missing
    # Keep only rows where open, high, low, close are not null and not empty strings
    df = df.replace('', np.nan)  # Replace empty strings with NaN
    df = df.dropna(subset=['open', 'high', 'low', 'close'])  # Drop rows with missing OHLC data
    
    # Optionally, you can also filter out rows where volume is 0 if that indicates invalid data
    # df = df[df['volume'] > 0]
    
    return df

def getOperation(file_path):
    '''
    timestamp,Decision,Hands
    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Operation file {file_path} does not exist.")
    
    df = pd.read_csv(file_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Clean operation data if needed
    df = df.replace('', np.nan)
    df = df.dropna()  # Remove rows with any missing values
    
    return df

def sync_data_by_date_range(df_market, df_operation):
    """
    Get date range from df_operation and sync with df_market data
    If date exists in df_market but not in df_operation, add it to df_operation with 0 values
    Returns filtered dataframes with common date range
    """
    # Ensure df_market is sorted by date (index)
    df_market = df_market.sort_index()


    # Get the date range from df_operation (start and end dates)
    operation_start_date = df_operation.index.min()
    operation_end_date = df_operation.index.max()
    
    print(f"Operation date range: {operation_start_date} to {operation_end_date}")
    
    # Filter df_market to the same date range
    df_market_filtered = df_market.loc[operation_start_date:operation_end_date]
    
    print(f"Market data entries in date range: {len(df_market_filtered)}")
    print(f"Operation data entries: {len(df_operation)}")
    
    # Reindex df_operation to match df_market_filtered dates, filling missing dates with 0
    df_operation_synced = df_operation.reindex(df_market_filtered.index, fill_value=0)
    
    # Keep the market data as is (already filtered to the date range)
    df_market_synced = df_market_filtered
    
    print(f"After synchronization:")
    print(f"Number of dates: {len(df_market_synced)}")
    print(f"Date range: {df_market_synced.index.min()} to {df_market_synced.index.max()}")
    
    return df_market_synced, df_operation_synced

TICKER = 'NIO'
df_extended = getStockMarketData(TICKER)
df_operation = getOperation('results/my-new-workspace-37256307_NIO.csv')
df_extended, df_operation = sync_data_by_date_range(df_extended, df_operation)

# Ensure high >= close >= low and high >= open >= low
for i in range(len(df_extended)):
    values = [df_extended.iloc[i]['open'], df_extended.iloc[i]['close']]
    df_extended.iloc[i, df_extended.columns.get_loc('high')] = max(values) + abs(np.random.normal(0, 100))
    df_extended.iloc[i, df_extended.columns.get_loc('low')] = min(values) - abs(np.random.normal(0, 100))

# Process the data
result = process_trading_data(df_extended)

# Display sample output
print("Sample Trading Signals Output:")
print(result.head(10).to_csv(index=True))
print(result.describe())

result.to_csv(f'results/rule_{TICKER}.csv')