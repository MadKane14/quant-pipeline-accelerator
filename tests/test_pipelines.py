import pandas as pd
import pytest
from src.pipeline_naive import run_pipeline_naive
from src.pipeline_numpy import run_pipeline_numpy
from src.pipeline_numba import run_pipeline_numba
from src.pipeline_parallel import run_pipeline_parallel

@pytest.fixture(scope="module")
def sample_data():
    df = pd.read_parquet("data/market_data_small.parquet")
    tickers = df['ticker'].unique()[:5]
    return df[df['ticker'].isin(tickers)].copy()

@pytest.fixture(scope="module")
def baseline_result(sample_data):
    return run_pipeline_naive(sample_data)

def test_numpy_pipeline_correctness(sample_data, baseline_result):
    numpy_result = run_pipeline_numpy(sample_data)
    assert numpy_result.shape == baseline_result.shape
    cols_to_check = ['daily_return', 'sma_20', 'volatility_20', 'lag_5']
    for col in cols_to_check:
        pd.testing.assert_series_equal(numpy_result[col], baseline_result[col], check_exact=False)

def test_numba_pipeline_correctness(sample_data, baseline_result):
    numba_result = run_pipeline_numba(sample_data)
    assert numba_result.shape == baseline_result.shape
    cols_to_check = ['daily_return', 'sma_20', 'volatility_20', 'lag_5']
    for col in cols_to_check:
        pd.testing.assert_series_equal(numba_result[col], baseline_result[col], rtol=1e-3)

def test_parallel_pipeline_correctness(sample_data, baseline_result):
    """Verifies that multiprocessing does not corrupt data order or mathematical outputs."""
    # We restrict n_jobs to 2 for the unit test to keep overhead low
    parallel_result = run_pipeline_parallel(sample_data, n_jobs=2)
    assert parallel_result.shape == baseline_result.shape
    cols_to_check = ['daily_return', 'sma_20', 'volatility_20', 'lag_5']
    for col in cols_to_check:
        pd.testing.assert_series_equal(parallel_result[col], baseline_result[col], rtol=1e-3)
