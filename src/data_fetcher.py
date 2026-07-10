import os
import argparse
import pandas as pd
import yfinance as yf

def fetch_sp500_tickers() -> list:
    url = "https://en.wikipedia.org/wiki/List_of_S&P_500_companies"
    # Bypass the 403 Forbidden error by spoofing a standard web browser
    tables = pd.read_html(
        url, 
        storage_options={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    df = tables[0]
    tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers

def build_market_dataset(start_date: str = "2010-01-01", end_date: str = "2025-12-31", output_path: str = None) -> pd.DataFrame:
    tickers = fetch_sp500_tickers()
    raw_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', threads=True)
    
    processed_dfs = []
    for ticker in tickers:
        if ticker in raw_data.columns.levels[0]:
            ticker_df = raw_data[ticker].dropna(how='all').copy()
            if not ticker_df.empty:
                ticker_df = ticker_df.reset_index()
                ticker_df['ticker'] = ticker
                processed_dfs.append(ticker_df)
                
    if not processed_dfs:
        raise ValueError("No data could be successfully downloaded.")
        
    full_df = pd.concat(processed_dfs, ignore_index=True)
    full_df.columns = [col.lower().replace(' ', '_') for col in full_df.columns]
    
    cols_order = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    full_df = full_df[cols_order]
    full_df = full_df.sort_values(by=['date', 'ticker']).reset_index(drop=True)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        full_df.to_parquet(output_path, index=False)
        
    return full_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, choices=['small', 'large'], default='small')
    args = parser.parse_args()
    
    if args.size == 'small':
        build_market_dataset(start_date="2023-01-01", end_date="2025-12-31", output_path="data/market_data_small.parquet")
    elif args.size == 'large':
        build_market_dataset(start_date="2010-01-01", end_date="2025-12-31", output_path="data/market_data_large.parquet")
