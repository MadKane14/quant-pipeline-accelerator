import pandas as pd
import cProfile
import pstats
import io
import time
from memory_profiler import profile
from src.pipeline_numba import run_pipeline_numba

@profile
def trace_memory(df):
    return run_pipeline_numba(df)

def main():
    print("Loading small dataset for profiling (~350k rows)...")
    df = pd.read_parquet("data/market_data_small.parquet")
    
    print("\n--- JIT COMPILATION WARM-UP ---")
    print("Running Numba for the first time (compiling to machine code)...")
    start_compile = time.time()
    # Pass a tiny 1,000-row slice just to trigger the compiler
    _ = run_pipeline_numba(df.head(1000).copy()) 
    print(f"Compilation finished in {time.time() - start_compile:.2f} seconds")
    
    print(f"\n--- MEMORY PROFILING ---")
    _ = trace_memory(df.copy())
    
    print(f"\n--- CPU PROFILING ---")
    pr = cProfile.Profile()
    
    start_time = time.time()
    pr.enable()
    _ = run_pipeline_numba(df.copy())
    pr.disable()
    end_time = time.time()
    
    print(f"Total Numba Execution Time: {end_time - start_time:.2f} seconds")
    
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(25)
    
    print(s.getvalue())

if __name__ == "__main__":
    main()
