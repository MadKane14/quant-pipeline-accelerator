import pandas as pd
import cProfile
import pstats
import io
import time
from memory_profiler import profile
from src.pipeline_numpy import run_pipeline_numpy

@profile
def trace_memory(df):
    return run_pipeline_numpy(df)

def main():
    print("Loading small dataset for profiling (~350k rows)...")
    df = pd.read_parquet("data/market_data_small.parquet")
    
    print(f"\n--- MEMORY PROFILING ---")
    print(f"Data shape: {df.shape}")
    _ = trace_memory(df.copy())
    
    print(f"\n--- CPU PROFILING ---")
    pr = cProfile.Profile()
    
    start_time = time.time()
    pr.enable()
    _ = run_pipeline_numpy(df.copy())
    pr.disable()
    end_time = time.time()
    
    print(f"Total NumPy Execution Time: {end_time - start_time:.2f} seconds")
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(25)
    
    print(s.getvalue())

if __name__ == "__main__":
    main()
