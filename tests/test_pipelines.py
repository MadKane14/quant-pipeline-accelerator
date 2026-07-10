import pandas as pd
import pytest
import numpy as np
from src.pipeline_naive import run_pipeline_naive

@pytest.fixture(scope="module")
def sample_data():
    """Loads a small slice of real data to keep test execution under 1 second."""
    df = pd.read_parquet("data/market_data_small.parquet")
    # Take only the first 5 tickers for unit testing
    tickers = df['ticker'].unique()[:5]
    return df[df['ticker'].isin(tickers)].copy()

def test_naive_pipeline_execution(sample_data):
    """Verifies the naive pipeline generates the correct columns and preserves row counts."""
    result = run_pipeline_naive(sample_data)
    
    expected_new_cols = ['daily_return', 'sma_20', 'volatility_20', 'lag_5']
    for col in expected_new_cols:
        assert col in result.columns
        
    assert len(result) == len(sample_data)
    
    # Check that rolling operations generated NaNs appropriately at the start of a ticker
    first_ticker = result['ticker'].iloc[0]
    first_ticker_data = result[result['ticker'] == first_ticker]
    assert pd.isna(first_ticker_data['sma_20'].iloc[0])
