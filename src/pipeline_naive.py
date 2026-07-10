import pandas as pd

def run_pipeline_naive(df: pd.DataFrame) -> pd.DataFrame:
    """
    Naive implementation of a quantitative feature engineering pipeline.
    Processes data ticker-by-ticker using a Python for-loop, appending results.
    This simulates standard, unoptimized Pandas code.
    """
    # Sort data chronologically per asset
    df = df.sort_values(['ticker', 'date']).copy()
    
    processed_chunks = []
    
    # Looping through groups manually instead of vectorizing
    for ticker in df['ticker'].unique():
        # Allocating new memory for a sliced copy
        chunk = df[df['ticker'] == ticker].copy()
        
        # 1. Daily return
        chunk['daily_return'] = chunk['close'].pct_change()
        
        # 2. Rolling 20-day Simple Moving Average (SMA)
        chunk['sma_20'] = chunk['close'].rolling(window=20).mean()
        
        # 3. Rolling 20-day Volatility (Standard Deviation of returns)
        chunk['volatility_20'] = chunk['daily_return'].rolling(window=20).std()
        
        # 4. Momentum (5-day lag)
        chunk['lag_5'] = chunk['close'].shift(5)
        
        processed_chunks.append(chunk)
        
    # Concatenating a massive list of DataFrames
    result_df = pd.concat(processed_chunks, ignore_index=True)
    
    return result_df
