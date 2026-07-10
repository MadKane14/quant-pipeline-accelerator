import pandas as pd
import cProfile
import pstats
import io
import time
from memory_profiler import profile
from src.pipeline_naive import run_pipeline_naive

# The @profile decorator will trace memory allocations during execution
@profile
def trace_memory(df):
    return run_pipeline_naive(df)

def main():
    print("Loading small dataset for profiling (~350k rows)...")
    # We use the small dataset; the naive loop on 1.5M rows would take way too long!
    df = pd.read_parquet("data/market_data_small.parquet")
    
    print(f"\n--- MEMORY PROFILING ---")
    print(f"Data shape: {df.shape}")
    _ = trace_memory(df.copy())
    
    print(f"\n--- CPU PROFILING ---")
    pr = cProfile.Profile()
    
    start_time = time.time()
    pr.enable()
    _ = run_pipeline_naive(df.copy())
    pr.disable()
    end_time = time.time()
    
    print(f"Total Naive Execution Time: {end_time - start_time:.2f} seconds")
    
    s = io.StringIO()
    # Sort by cumulative time to expose the heaviest functions
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(25)  # Print the top 25 slowest calls
    
    print(s.getvalue())

if __name__ == "__main__":
    main()
