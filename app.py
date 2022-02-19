from ast import And
import streamlit as st

import requests
import pandas as pd
import hvplot.pandas
import panel as pn
import json
import numpy as np
import param
import alpaca_trade_api as tradeapi


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
  roi_low, roi_high = roi

  main_df = get_sector_data(sector)
  
  main_df = filter(main_df, beta_low,beta_high,sharpe_low, sharpe_high,roi_low, roi_high)

  if len(main_df.columns) >0:
      st.line_chart(main_df)
  else:
      st.write("No stocks matched the criteria selected")


  #st.write(beta_low, beta_high, sharpe_low, sharpe_high, roi_low, roi_high)

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

#EH:  Calculate daily return, ROI, std, sharpe ratio
def cal_ratio(close_price_df):
  #EH: daily rate
  daily_return_df=close_price_df.pct_change().dropna()
  #EH: cumulative return
  cumulative_return_df=((daily_return_df+1).cumprod()-1).dropna()
  #EH:  get latest cumulative return value for filter purpose
  roi_df=cumulative_return_df.iloc[-1]        
  #EH:  daily_return_mean_df
  daily_return_mean_df=daily_return_df.mean()
  #EH: std
  std_df=daily_return_df.std()
  #EH: ROI
  # annualized_return
  trade_days=252
  annualized_return=daily_return_mean_df*trade_days
  #EH: annualized std
  annualized_std=std_df * (trade_days ** 1/2)
  #EH: sharpe ratio
  sharpe_df=annualized_return/annualized_std
        
  return roi_df, sharpe_df, std_df

#EH:  filter for sharpe and roi
def filter(main_df, beta_low,beta_high,sharpe_low, sharpe_high, roi_low, roi_high):

    roi_df, sharpe_df, std_df = cal_ratio(main_df)

    for ticker in roi_df.keys():
        if ticker in main_df and roi_df[ticker] < roi_low or roi_df[ticker] > roi_high:
            #st.write(">> dropping:", ticker, "::",roi_df[ticker], ":::", roi_low, roi_high)
            main_df.drop(columns=[ticker], axis=1,inplace=True)
    
    for ticker in sharpe_df.keys():
        if ticker in main_df and sharpe_df[ticker] < sharpe_low or sharpe_df[ticker] > sharpe_high:
            #st.write(">> dropping:", ticker, "::",sharpe_df[ticker], ":::", sharpe_low, sharpe_high)
            main_df.drop(columns=[ticker], axis=1,inplace=True)

    return main_df

#EH:  caluclate confidence interval
ci_zscore_dict={'99%':2.576,
                '95%':1.96}

#EH:  set blank selected tickers
selected_tickers=[]

#EH:  define function to print confidence interval and its retuns
def confidence(stock,conf_pct):
    downside=daily_return_mean_df[stock] - ci_zscore_dict[conf_pct] *std_df[stock]
    upside=daily_return_mean_df[stock] + ci_zscore_dict[conf_pct] *std_df[stock]
    print(f"Using a {conf_pct} confidence interval, "
      f"the {stock} could trade down as much as {(downside * 100): .4f}%, "
      f"and up as much as {(upside * 100): .4f}%.")


    #EH: print CI & its returns for selected tickers
    num_of_stock=len(selected_tickers)
    for num in range(num_of_stock):
        confidence(selected_stock[num],'99%')
        confidence(selected_stock[num],'95%')



def main():

  # Title
  st.title("Stock Selector App")

  #st.sidebar.title("Controls")
  st.sidebar.info( "Select the criteria that you want here.")
  sector = st.sidebar.selectbox("Sectors", sectors)
  beta = st.sidebar.slider('Beta Range', 0.0, 5.0, (1.0,4.0))
  sharpe = st.sidebar.slider('Sharpe Range', 0.0, 2.0, (0.0,2.0))
  roi = st.sidebar.slider('ROI Range', 0.0, 5.0, (0.0,5.0))    
  
  execute(sector, beta, sharpe, roi)
  

main()  

