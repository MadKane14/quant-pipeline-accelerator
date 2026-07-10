


# High-Performance Python: Quantitative Data Pipeline Case Study

A systems-engineering approach to optimizing a CPU- and memory-intensive Python data pipeline. 

This project explores the architectural trade-offs of processing large-scale financial time-series data using standard Python, C-level vectorization, Just-In-Time (JIT) compilation, and multi-core parallelism. The goal is to calculate common quantitative predictive features (rolling volatility, momentum lags, moving averages) across millions of rows while strictly minimizing RAM usage and execution time.

## Final Benchmark Results

Tested on an authentic market dataset of **~1.9 Million rows** (Historical S&P 500 Daily OHLCV data).

| Pipeline Type | Runtime (s) | Speedup | Throughput (Rows/Sec) | Peak Mem (MiB) |
| :--- | :--- | :--- | :--- | :--- |
| **Naive (Baseline)** | 4.328 | 1.0x | 437,765 | 410.9 |
| **NumPy (Vectorized)** | 0.472 | 9.2x | 4,017,562 | 224.2 |
| **Numba (JIT Compiled)**| **0.456** | **9.5x** | **4,153,672** | **209.7** |
| **Parallel (Joblib)** | 3.027 | 1.4x | 625,961 | 569.9 |

## Systems Engineering Insights

This project highlights several critical realities of high-performance computing in Python:

1. **The Cost of Python Loops (The Naive Bottleneck):** The baseline implementation suffered from massive overhead due to the Python interpreter's interaction with Pandas data structures. Profiling revealed that mathematical calculations took almost zero time; 95%+ of the CPU cycles were wasted on `__setitem__`, memory allocation for intermediate `.copy()` operations, and Pandas slicing.
2. **Vectorization vs. Memory Footprint:** Pushing operations down to C via NumPy yielded a **9.2x speedup**. However, vectorization requires allocating massive contiguous blocks of memory simultaneously. While it eliminated the "garbage" allocations of the naive loop, it still required 224 MiB of peak RAM to hold the arrays during computation.
3. **The Power of JIT Compilation :** By extracting the mathematical kernels and compiling them directly to machine code using `Numba`, we achieved the highest throughput (**4.15M rows/sec**) and the lowest memory footprint (**209 MiB**). Numba operates strictly on raw NumPy arrays without intermediate Pandas overhead, hitting the physical lower bound of memory required to store the results.
4. **The IPC Serialization overhead:** Distributing the workload across multiple CPU cores via `joblib` resulted in a surprisingly slow 3.02s runtime and the highest memory footprint (569.9 MiB). This perfectly demonstrates the **Inter-Process Communication (IPC) bottleneck**. The time required to serialize (pickle) the data, move it across the OS to worker cores, and deserialize the results drastically outweighed the computational time of the math itself. Parallelism is not free; data movement is expensive.

## Tech Stack
* **Python 3**
* **Pandas / NumPy** (Data manipulation and C-level vectorization)
* **Numba** (LLVM JIT Compiler for machine-code optimization)
* **Joblib** (Multiprocessing and worker pool management)
* **cProfile / memory_profiler / tracemalloc** (Algorithmic profiling)
* **Pytest** (Automated correctness validation)
* **yfinance / pyarrow** (Real-world data ingestion and Parquet storage)

## How to Reproduce This Case Study

### 1. Environment Setup
Clone the repository and set up an isolated virtual environment:
```bash
git clone [https://github.com/YOUR_USERNAME/python-optimization-case-study.git](https://github.com/YOUR_USERNAME/python-optimization-case-study.git)
cd python-optimization-case-study
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt


### 2. Data Acquisition

Fetch the real-world historical market datasets.

```bash
# Downloads a smaller ~350k row dataset for testing
python3 src/data_fetcher.py --size small

# Downloads the massive ~1.9M row dataset for benchmarking
python3 src/data_fetcher.py --size large

```

### 3. Run Automated Tests

Ensure all optimized pipelines mathematically match the baseline without corrupting row order or statistical accuracy.

```bash
pytest tests/test_pipelines.py

```

### 4. Execute the Profiler & Benchmarks

Run the line-by-line memory tracer and CPU profiler on the small dataset:

```bash
python3 -m benchmarks.run_profiler

```

Run the master benchmark on the large dataset to generate the final metrics table:

```bash
python3 -m benchmarks.run_benchmarks

```