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
from MCForecastTools import MCSimulation
from numpy import random

# FIXME! Need to move these somewhere else.
alpaca_key = 'PKQGP0BR4BOGDYH6946H'
alpaca_secret = 'dNcBOKDiV3Y9mrAj81rCkPT2uysP6my2ZNz6bBHy'

alpaca = tradeapi.REST(alpaca_key, alpaca_secret, api_version='v2')

sectors_to_tickers = {}
with open("C:/Users/eunic/OneDrive/Desktop/stock-selector/sp_500_sectors_to_ticker.json") as json_file:
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
    get_beta(main_df)

    main_df = filter(main_df, beta_low, beta_high, sharpe_low,
                     sharpe_high, roi_low, roi_high)

    # EH:  call series data
    roi_s, sharpe_s, std_s = cal_ratio(main_df)
    beta_s = get_beta(main_df)
    # EH:  stats dataframe for streamlit display
    df_roi = pd.DataFrame(roi_s)
    df_roi.columns = ['ROI']
    df_sharpe = pd.DataFrame(sharpe_s)
    df_sharpe.columns = ['Sharpe']
    df_std = pd.DataFrame(std_s)
    df_std.columns = ['STD']
    # df_beta=pd.DataFrame(beta_s)
    # df_beta.columns=['Beta']
    stats_df = pd.concat([df_roi, df_sharpe, df_std], axis=1)
    # df_beta

    if len(main_df.columns) > 0:
        st.line_chart(main_df)

        # EH: streamlit dataframe display
        st.subheader('Daily Closing Price')
        st.dataframe(main_df.style.highlight_max(axis=0))
        # EH: rates display on streamlit
        st.subheader('Rates')
        st.dataframe(stats_df.style.highlight_max(axis=0))
        # EH:  stock selection for MC simulation
        st.subheader('Please select up to 4 stocks for MC simulation.')

        # stock selection widget to get weight%
        # multi_select_stock = pn.widgets.MultiSelect(name='Stock Selections', value=[list(df_roi.index)[0]],options=list(df_roi.index), size=10)
        selected_stock = st.multiselect("Tickers:", list(df_roi.index))

        # EH: get weight% for MC simulation of 4 stocks
        weight_dict = {}
        if len(selected_stock) > 4:
            st.error(
                'Invalid selection count.  Please select up to 4 stocks for MC simulation.')

        else:
            # EH:  create dictionary for weight % per ticker
        
            for each in selected_stock:
                number = st.number_input(
                    f'Please provide a weight percentage for {each}.',min_value=0)
                st.write(f'The current {each} weight percentage is ', number)
                weight_dict[each] = number
        
        sum_weight_pct = sum(weight_dict.values())

        # EH:  error message for weight percent <>100.
        if sum_weight_pct != 100:
            st.error(
                    'Invalid weight percentage input.  The sum of weight percentage should be 100.')
        else:
            st.write('Thank you for the input!')

    else:
        st.write("No stocks matched the criteria selected")


def get_beta(main_df):
    daily_returns = main_df.pct_change().dropna()
    daily_returns.index = pd.to_datetime(daily_returns.index)
    # st.write(daily_returns)

    # Get SPY Dataframe
    start = (pd.Timestamp.now() - pd.Timedelta(days=365)).isoformat()
    end = pd.Timestamp.now().isoformat()
    spy_df = alpaca.get_barset(
        'SPY', start=start, end=end, timeframe='1D', limit=351).df
    spy_df = spy_df['SPY'].drop(columns=['open', 'high', 'low', 'volume'])
    spy_df_daily_returns = spy_df.pct_change().dropna()
    spy_var = spy_df_daily_returns['close'].rolling(window=252).var()

    # calculate cov and beta
    tickers_list = []
    beta_list = []
    for ticker in tickers_list:
        cov = daily_returns[ticker].cov(spy_df_daily_returns['close'])
        beta = (cov/spy_var).mean()
        beta_list.append({ticker: beta})
    beta_df = pd.DataFrame(beta_list)
    beta_df = beta_df.stack()
    beta_df = beta_df.reset_index(level=0)
    beta_df = beta_df.drop(columns='level_0')

    return beta_df


def get_sector_data(sector):
    """
    This function is responsible for loading all of the stock data within a sector.
    """

    stocks_to_load = sectors_to_tickers[sector]
    start = (pd.Timestamp.now() - pd.Timedelta(days=365)).isoformat()
    end = pd.Timestamp.now().isoformat()

    main_df = alpaca.get_barset(
        stocks_to_load, start=start, end=end, timeframe='1D', limit=352).df

    # dropping unused columns
    main_df.drop(columns=['open', 'high', 'low', 'volume'],
                 axis=1, level=1, inplace=True)

    # the y axis is multi-dementional, and this is flattening it.
    main_df.columns = [col[0] for col in main_df.columns.values]

    return main_df

# EH:  Calculate daily return, ROI, std, sharpe ratio


def cal_ratio(close_price_df):
    # EH: daily rate
    daily_return_df = close_price_df.pct_change().dropna()
    # EH: cumulative return
    cumulative_return_df = ((daily_return_df+1).cumprod()-1).dropna()
    # EH:  get latest cumulative return value for filter purpose
    roi_df = cumulative_return_df.iloc[-1]
    #EH:  daily_return_mean_df
    daily_return_mean_df = daily_return_df.mean()
    #EH: std
    std_df = daily_return_df.std()
    #EH: ROI
    # annualized_return
    trade_days = 252
    annualized_return = daily_return_mean_df*trade_days
    # EH: annualized std
    annualized_std = std_df * (trade_days ** 1/2)
    # EH: sharpe ratio
    sharpe_df = annualized_return/annualized_std

    return roi_df, sharpe_df, std_df

# EH:  filter for sharpe and roi


def filter(main_df, beta_low, beta_high, sharpe_low, sharpe_high, roi_low, roi_high):

    roi_df, sharpe_df, std_df = cal_ratio(main_df)

    for ticker in roi_df.keys():
        if ticker in main_df and roi_df[ticker] < roi_low or roi_df[ticker] > roi_high:
            #st.write(">> dropping:", ticker, "::",roi_df[ticker], ":::", roi_low, roi_high)
            main_df.drop(columns=[ticker], axis=1, inplace=True)

    for ticker in sharpe_df.keys():
        if ticker in main_df and sharpe_df[ticker] < sharpe_low or sharpe_df[ticker] > sharpe_high:
            #st.write(">> dropping:", ticker, "::",sharpe_df[ticker], ":::", sharpe_low, sharpe_high)
            main_df.drop(columns=[ticker], axis=1, inplace=True)

    return main_df


# EH:  caluclate confidence interval
ci_zscore_dict = {'99%': 2.576,
                  '95%': 1.96}

# EH:  set blank selected tickers
selected_tickers = []

# EH:  define function to print confidence interval and its retuns


def confidence(stock, conf_pct):
    downside = daily_return_mean_df[stock] - \
        ci_zscore_dict[conf_pct] * std_df[stock]
    upside = daily_return_mean_df[stock] + \
        ci_zscore_dict[conf_pct] * std_df[stock]
    print(f"Using a {conf_pct} confidence interval, "
          f"the {stock} could trade down as much as {(downside * 100): .4f}%, "
          f"and up as much as {(upside * 100): .4f}%.")

    # EH: print CI & its returns for selected tickers
    num_of_stock = len(selected_tickers)
    for num in range(num_of_stock):
        confidence(selected_stock[num], '99%')
        confidence(selected_stock[num], '95%')

    # the y axis is multi-dementional, and this is flattening it.
    main_df.columns = [col[0] for col in main_df.columns.values]

    return main_df


def main():

    # Title
    st.title("Stock Selector App")

    # st.sidebar.title("Controls")
    st.sidebar.info("Select the criteria that you want here.")
    sector = st.sidebar.selectbox("Sectors", sectors)
    beta = st.sidebar.slider('Beta Range', 0.0, 5.0, (1.0, 4.0))
    sharpe = st.sidebar.slider('Sharpe Range', 0.0, 2.0, (0.0, 2.0))
    roi = st.sidebar.slider('ROI Range', 0.0, 5.0, (0.0, 5.0))

    execute(sector, beta, sharpe, roi)


main()

# Configure a Monte Carlo simulation to forecast five years cumulative returns


def mc_simulation(main_df):

    stocks_to_load = sectors_to_tickers[sector]
    stocks_count = stocks_to_load.count()
    weight = np.random.rand(stocks_count)
    weight /= weight.sum()
    num_simulation = 1000

    MC_fiveyear = MCSimulation(
        portfolio_data=main_df,
        weights=weights,
        num_simulation=num_simulation,
        num_trading_days=252*5
    )
    return MC_fiveyear.portfolio_data

# mc_simulation(main_df)
