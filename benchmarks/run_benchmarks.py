import pandas as pd
import time
import tracemalloc
from src.pipeline_naive import run_pipeline_naive
from src.pipeline_numpy import run_pipeline_numpy
from src.pipeline_numba import run_pipeline_numba
from src.pipeline_parallel import run_pipeline_parallel

def measure_pipeline(name, func, df):
    """Helper function to cleanly measure time, memory, and throughput."""
    print(f"Running {name} Pipeline...")
    
    # Start the Python memory tracker
    tracemalloc.start()
    start_time = time.time()
    
    # Execute the pipeline
    _ = func(df.copy())
    
    # Stop timers and trackers
    runtime = time.time() - start_time
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Calculate metrics
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)
    rows_per_second = len(df) / runtime
    
    print(f"   Done in {runtime:.2f}s | Peak Mem: {peak_mem_mb:.1f} MiB")
    
    return {
        'name': name,
        'runtime': runtime,
        'throughput': rows_per_second,
        'memory_mb': peak_mem_mb
    }

def run_benchmark():
    print("Loading massive dataset for final benchmark (~1.5M+ rows)...")
    df = pd.read_parquet("data/market_data_large.parquet")
    n_rows = len(df)
    print(f"Dataset Shape: {df.shape}\n")
    
    print("Warming up Numba and Parallel environments...")
    warmup_df = df.head(5000).copy()
    _ = run_pipeline_numba(warmup_df)
    _ = run_pipeline_parallel(warmup_df, n_jobs=2)
    print("Warm-up complete.\n")
    
    # Execute all benchmarks
    results = []
    results.append(measure_pipeline("Naive", run_pipeline_naive, df))
    results.append(measure_pipeline("NumPy", run_pipeline_numpy, df))
    results.append(measure_pipeline("Numba", run_pipeline_numba, df))
    results.append(measure_pipeline("Parallel", run_pipeline_parallel, df))
    
    # Generate the Multi-Metric Summary Table
    print("\n" + "="*85)
    print(f"{'Pipeline Type':<15} | {'Runtime (s)':<12} | {'Speedup':<10} | {'Rows/Sec':<15} | {'Peak Mem (MiB)':<15}")
    print("-" * 85)
    
    baseline_time = results[0]['runtime']
    
    for res in results:
        speedup = baseline_time / res['runtime']
        print(f"{res['name']:<15} | {res['runtime']:<12.3f} | {speedup:<10.1f} | {res['throughput']:<15,.0f} | {res['memory_mb']:<15.1f}")
    print("="*85 + "\n")

if __name__ == "__main__":
    run_benchmark()
