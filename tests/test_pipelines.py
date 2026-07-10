import pandas as pd
import pytest
from src.pipeline_naive import run_pipeline_naive
from src.pipeline_numpy import run_pipeline_numpy

@pytest.fixture(scope="module")
def sample_data():
    df = pd.read_parquet("data/market_data_small.parquet")
    tickers = df['ticker'].unique()[:5]
    return df[df['ticker'].isin(tickers)].copy()

@pytest.fixture(scope="module")
def baseline_result(sample_data):
    """Generate the 'source of truth' from the naive pipeline."""
    return run_pipeline_naive(sample_data)

def test_numpy_pipeline_correctness(sample_data, baseline_result):
    """Verifies the optimized numpy pipeline matches the naive baseline exactly."""
    numpy_result = run_pipeline_numpy(sample_data)
    
    # Check shape
    assert numpy_result.shape == baseline_result.shape
    
    # Check that all calculated columns are numerically identical
    cols_to_check = ['daily_return', 'sma_20', 'volatility_20', 'lag_5']
    for col in cols_to_check:
        # We use check_exact=False to allow for tiny floating point differences, 
        # but the logic should be effectively identical.
        pd.testing.assert_series_equal(
            numpy_result[col], 
            baseline_result[col], 
            check_exact=False
        )
