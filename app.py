import streamlit as st

import requests
import pandas as pd
import hvplot.pandas
import panel as pn
import json
import numpy as np
import param
import alpaca_trade_api as tradeapi
from MCForecastTools import MCSimulation


# FIXME! Need to move these somewhere else.
alpaca_key = 'PKQGP0BR4BOGDYH6946H'
alpaca_secret = 'dNcBOKDiV3Y9mrAj81rCkPT2uysP6my2ZNz6bBHy'

alpaca = tradeapi.REST(alpaca_key,alpaca_secret,api_version='v2')

sectors_to_tickers = {}
with open('sp_500_sectors_to_ticker.json') as json_file:
    sectors_to_tickers = json.load(json_file)
sectors = list(sectors_to_tickers.keys())

def execute(sector, beta, sharpe, roi):
    """
    This is the main data gathering for this app. It will call other functions
    to assemble a main dataframe which can be used in different ways.
    """

    # unpacking the lows and highs into variables that can be used more easliy.
    beta_low, beta_high = beta
    sharpe_low, sharpe_high = sharpe
    roi_low, roi_high = beta

    main_df = get_sector_data(sector)
    get_beta(main_df)
    st.line_chart(main_df)  
    
def get_beta(main_df):
  daily_returns = main_df.pct_change().dropna()
  st.write(daily_returns)
  spy_df = alpaca.get_barset('SPY', timeframe ='1D').df
  spy_df = spy_df['SPY'].drop(columns=['open', 'high', 'low', 'volume'])
  spy_df_daily_returns = spy_df.pct_change().dropna()
  spy_var = spy_df_daily_returns['close'].rolling(window=252).var()
  cov = daily_returns[:].cov(spy_df_daily_returns['close'])
  beta = cov/spy_var
  return beta


def get_sector_data(sector):
    """
    This function is responsible for loading all of the stock data within a sector.
    """
  
    stocks_to_load = sectors_to_tickers[sector]
    start = (pd.Timestamp.now() - pd.Timedelta(days=365)).isoformat()
    end = pd.Timestamp.now().isoformat()
  
    main_df = alpaca.get_barset(stocks_to_load, start=start, end=end, timeframe='1D', limit=352).df
       
    #dropping unused columns
    main_df.drop(columns=['open','high','low','volume'], axis=1, level=1,inplace=True)
    
    # the y axis is multi-dementional, and this is flattening it.
    main_df.columns = [col[0] for col in main_df.columns.values]

    return main_df

  



def main():

    # Title
    st.title("Stock Selector App")

    #st.sidebar.title("Controls")
    st.sidebar.info( "Select the criteria that you want here.")
    sector = st.sidebar.selectbox("Sectors", sectors)
    beta = st.sidebar.slider('Beta Range', 0.0, 5.0, (1.0,4.0))
    sharpe = st.sidebar.slider('Sharpe Range', 0.0, 5.0, (1.0,4.0))
    roi = st.sidebar.slider('ROI Range', 0.0, 5.0, (1.0,4.0))    
  
    execute(sector, beta, sharpe, roi)
  

main()  
