import pandas as pd
import numpy as np
from numba import njit

@njit
def compute_rolling_volatility_numba(returns_array: np.ndarray, group_sizes: np.ndarray, window: int) -> np.ndarray:
    """
    A highly optimized, pre-compiled C-level loop for calculating rolling volatility.
    Operates strictly on raw numpy arrays for zero memory overhead.
    """
    n = len(returns_array)
    out = np.empty(n, dtype=np.float64)
    out[:] = np.nan
    
    current_idx = 0
    
    # Mathematical adjustment to match Pandas' ddof=1 (Sample Standard Deviation)
    ddof_adjustment = np.sqrt(window / (window - 1.0))
    
    for size in group_sizes:
        start_idx = current_idx
        end_idx = current_idx + size
        
        for i in range(start_idx, end_idx):
            if i >= start_idx + window - 1:
                window_slice = returns_array[i - window + 1 : i + 1]
                # Apply the adjustment factor to the numpy output
                out[i] = np.std(window_slice) * ddof_adjustment
                
        current_idx = end_idx
        
    return out

def run_pipeline_numba(df: pd.DataFrame) -> pd.DataFrame:
    """
    Integrates the compiled Numba math kernel back into the Pandas pipeline.
    """
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    df['daily_return'] = df.groupby('ticker')['close'].pct_change()
    df['lag_5'] = df.groupby('ticker')['close'].shift(5)
    df['sma_20'] = df.groupby('ticker')['close'].rolling(window=20).mean().reset_index(level=0, drop=True)
    
    group_sizes = df.groupby('ticker').size().values
    returns_array = df['daily_return'].values
    
    df['volatility_20'] = compute_rolling_volatility_numba(returns_array, group_sizes, window=20)
    
    return df
