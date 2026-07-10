import os
import pandas as pd
from src.data_fetcher import fetch_sp500_tickers, build_market_dataset

def test_fetch_tickers():
    tickers = fetch_sp500_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) > 400  
    assert "AAPL" in tickers

def test_market_dataset_structure():
    df = build_market_dataset(start_date="2025-12-01", end_date="2025-12-10")
    
    expected_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in expected_columns)
    assert len(df) > 0
    
    assert pd.api.types.is_datetime64_any_dtype(df['date'])
    # Updated to correctly validate modern pandas string types
    assert pd.api.types.is_string_dtype(df['ticker']) or pd.api.types.is_object_dtype(df['ticker'])
