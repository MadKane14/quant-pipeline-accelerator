import pandas as pd

def run_pipeline_numpy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimized implementation using vectorized Pandas/NumPy operations.
    Eliminates Python-level for-loops and intermediate memory copies.
    """
    # Fix: Added reset_index to ensure output structure exactly matches the naive baseline
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # 1. Daily return (vectorized groupby)
    df['daily_return'] = df.groupby('ticker')['close'].pct_change()
    
    # 2. Momentum (5-day lag, vectorized groupby)
    df['lag_5'] = df.groupby('ticker')['close'].shift(5)
    
    # 3. Rolling 20-day SMA 
    df['sma_20'] = df.groupby('ticker')['close'].rolling(window=20).mean().reset_index(level=0, drop=True)
    
    # 4. Rolling 20-day Volatility
    df['volatility_20'] = df.groupby('ticker')['daily_return'].rolling(window=20).std().reset_index(level=0, drop=True)
    
    return df
