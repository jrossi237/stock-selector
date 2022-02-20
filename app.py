from ast import And
import streamlit as st
import os
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
from pathlib import Path
from PIL import Image

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
    
    get_beta(main_df)
#RA: Temporarily commented to facilitate montecarlo simulation - need to discuss alternatives to make available multilevel indexes
    #if len(main_df.columns) >0:
     #   st.line_chart(main_df)
    #else:
     #   st.write("No stocks matched the criteria selected")
    
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
    #RA: Inserted global varaible (as we will require multi-demensional df for MC analysis
    global closing_prices_df
    stocks_to_load = sectors_to_tickers[sector]
    start = (pd.Timestamp.now() - pd.Timedelta(days=365)).isoformat()
    end = pd.Timestamp.now().isoformat()
  
    main_df = alpaca.get_barset(stocks_to_load, start=start, end=end, timeframe='1D', limit=352).df
       
    #dropping unused columns
    main_df.drop(columns=['open','high','low','volume'], axis=1, level=1,inplace=True)
    
    #RA: Removing Timestamp
    main_df.index = main_df.index.date   
    main_df.index.name = 'date'
    #RA: Inserted global varaible & created a copy of main_df(as we will require multi-demensional df for MC analysis
    closing_prices_df = pd.DataFrame(main_df)
    # the y axis is multi-dementional, and this is flattening it.
#RA: Temporarily commented to facilitate montecarlo simulation - need to discuss alternatives to make available multilevel indexes
    #main_df.columns = [col[0] for col in main_df.columns.values]
    #closing_prices_df.columns = pd.MultiIndex.from_product([closing_prices_df.columns, ['closing']])
    return main_df

def print_closing_prices(closing_prices_df):
    st.write(closing_prices_df)
    
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
    #RA set the num_of_stock as a global varaible
    global num_of_stock
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


    # the y axis is multi-dementional, and this is flattening it.
    main_df.columns = [col[0] for col in main_df.columns.values]

    return main_df

# RA: Configure a Monte Carlo simulation to forecast five years cumulative returns
def mc(closing_prices_df):
    global MC_fiveyear
    weight = np.random.rand(10)
    weight /=weight.sum()
    MC_fiveyear = MCSimulation(
        portfolio_data = closing_prices_df,
        weights = weight,
        num_simulation = 500,
        num_trading_days = 500
    )
    MC_fiveyear.portfolio_data
    MC_fiveyear.calc_cumulative_return()
    # Plot simulation outcomes
    st.write(MC_fiveyear.portfolio_data)
    MC_sim_line_plot = MC_fiveyear.plot_simulation()
    MC_sim_line_plot.get_figure()
    # Save the plot for future use
    MC_sim_line_plot.get_figure().savefig("MC_fiveyear_sim_plot.png", bbox_inches="tight")
    img_path = Path("/Users/unicorn/Desktop/stock-selector/MC_fiveyear_sim_plot.png")
    image = Image.open(img_path)
    st.image(image, caption='Monte Carlo Simulation')
    # Plot probability distribution and confidence intervals
    MC_sim_dist_plot = MC_fiveyear.plot_distribution()
    MC_sim_dist_plot.get_figure()
    # Save the plot for future use
    MC_sim_dist_plot.get_figure().savefig("MC_fiveyear_dist_plot.png", bbox_inches="tight", rot =90)
    img1_path = Path("/Users/unicorn/Desktop/stock-selector/MC_fiveyear_dist_plot.png")
    image = Image.open(img1_path)
    st.image(image, caption='Monte Carlo Distribution')
    MC_summary_statistics = MC_fiveyear.summarize_cumulative_return()
    st.table(MC_summary_statistics) 
    #return MC_fiveyear
    
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
    #RA Montecarlo simulation
    print_closing_prices(closing_prices_df)
    #RA Montecarlo simulation
    mc(closing_prices_df)
    
main()  

