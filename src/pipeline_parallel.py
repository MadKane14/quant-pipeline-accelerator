import pandas as pd
import numpy as np
import multiprocessing
from joblib import Parallel, delayed
from src.pipeline_numba import run_pipeline_numba

def run_pipeline_parallel(df: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    """
    Scales the optimized pipeline across multiple CPU cores by chunking the dataset.
    """
    # -1 tells joblib to use all available CPU cores
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()
        
    tickers = df['ticker'].unique()
    
    # Split the unique tickers into N chunks (one chunk per CPU core)
    ticker_chunks = np.array_split(tickers, n_jobs)
    
    def process_chunk(ticker_chunk):
        """Helper function to process a specific chunk of data."""
        # Filter the dataframe for this specific chunk of tickers
        chunk_df = df[df['ticker'].isin(ticker_chunk)].copy()
        # Run our fastest single-core pipeline on this chunk
        return run_pipeline_numba(chunk_df)

    # Distribute the chunks across the cores concurrently
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_chunk)(chunk) for chunk in ticker_chunks if len(chunk) > 0
    )
    
    # Recombine the processed chunks and ensure the final structure matches our baseline
    final_df = pd.concat(results, ignore_index=True)
    return final_df.sort_values(['ticker', 'date']).reset_index(drop=True)
