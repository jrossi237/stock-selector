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
from numpy import random

# FIXME! Need to move these somewhere else.
alpaca_key = 'PKQGP0BR4BOGDYH6946H'
alpaca_secret = 'dNcBOKDiV3Y9mrAj81rCkPT2uysP6my2ZNz6bBHy'

alpaca = tradeapi.REST(alpaca_key,alpaca_secret,api_version='v2')

sectors_to_tickers = {}
with open('sp_500_sectors_to_ticker.json') as json_file:
    sectors_to_tickers = json.load(json_file)
sectors = list(sectors_to_tickers.keys())

styles = """
body {
    color: red;
    background-color: #4F8BF9;
}

.stTextInput>div>div>input {
    color: #4F8BF9;
}
"""

#st.markdown(f'<style>{styles}</style>', unsafe_allow_html=True)

foo =  """
    <style>

    [data-testid="stVerticalBlock"] {
        width: 100px;
        background-color: red !important;
    }
    [data-testid="stHorizontalBlock"][aria-expanded="true"] > div:first-child {
        width: 100px;
        background-color: red !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """

#st.markdown(foo, unsafe_allow_html=True)

def execute(sector, beta, sharpe, roi):
    """
    This is the main data gathering for this app. It will call other functions
    to assemble a main dataframe which can be used in different ways.
    """
    top_panel = st.container()
    bottom_panel = st.container()    
    
    # unpacking the lows and highs into variables that can be used more easliy.
    beta_low, beta_high = beta
    sharpe_low, sharpe_high = sharpe
    roi_low, roi_high = beta

    main_df = get_sector_data(sector)
    get_beta(main_df)
    with top_panel:
        st.line_chart(main_df)

    with bottom_panel:
        st.header("Further analysis")
        tickers = main_df.keys()
      
        for ticker in tickers:
            text_field(ticker, [1,2])
            #st.text_input(ticker)
            pass
        
def text_field(label, columns=None, **input_params):
    c1, c2 = st.columns(columns or [1, 4])

    # Display field name with some alignment
    c1.markdown("##")
    c1.markdown(label)

    # Sets a default key parameter to avoid duplicate key errors
    input_params.setdefault("key", label)

    # Forward text input parameters
    return c2.text_input("", **input_params)    
    
def get_beta(main_df):
    daily_returns = main_df.pct_change().dropna()
    daily_returns.index = pd.to_datetime(daily_returns.index)
    #st.write(daily_returns)
    
    #Get SPY Dataframe
    start = (pd.Timestamp.now() - pd.Timedelta(days=365)).isoformat()
    end = pd.Timestamp.now().isoformat()
    spy_df = alpaca.get_barset('SPY', start=start, end=end, timeframe='1D', limit=351).df
    spy_df = spy_df['SPY'].drop(columns=['open', 'high', 'low', 'volume'])
    spy_df_daily_returns = spy_df.pct_change().dropna()
    spy_var = spy_df_daily_returns['close'].rolling(window=252).var()
    
    #calculate cov and beta
    tickers_list =[]
    beta_list = []
    for ticker in tickers_list:
        cov = daily_returns[ticker].cov(spy_df_daily_returns['close'])
        beta = (cov/spy_var).mean()
        beta_list.append({ticker:beta})
    beta_df = pd.DataFrame(beta_list)
    beta_df = beta_df.stack()
    beta_df = beta_df.reset_index(level=0)
    beta_df = beta_df.drop(columns = 'level_0')
        
    return beta_df


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

#main_df (count)


# Configure a Monte Carlo simulation to forecast five years cumulative returns
def mc_simulation(main_df):

    stocks_to_load = sectors_to_tickers[sector]
    stocks_count = stocks_to_load.count()
    weight = np.random.rand(stocks_count)
    weight /=weight.sum()
    num_simulation = 1000
    
    MC_fiveyear = MCSimulation(
        portfolio_data = main_df,
        weights = weights,
        num_simulation = num_simulation,
        num_trading_days = 252*5
    )
    return MC_fiveyear.portfolio_data

#mc_simulation(main_df)

