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
import seaborn as sns
import matplotlib.pyplot as plt
from services.AlpacaService import AlpacaService

# The wide option will take up the entire screen.
st.set_page_config(page_title="Stock Selector App",layout="wide")
# this is change the page so that it will take a max with of 1200px, instead
# of the whole screen.
st.markdown(
        f"""<style>.main .block-container{{ max-width: 1200px }} </style> """,
        unsafe_allow_html=True,
)

# FIXME! Need to move these somewhere else.
alpaca_key = 'PKQGP0BR4BOGDYH6946H'
alpaca_secret = 'dNcBOKDiV3Y9mrAj81rCkPT2uysP6my2ZNz6bBHy'

alpaca = tradeapi.REST(alpaca_key, alpaca_secret, api_version='v2')

alpacaService = AlpacaService(alpaca_key, alpaca_secret)

sectors_to_tickers = {}
with open("sp_500_sectors_to_ticker.json") as json_file:
    sectors_to_tickers = json.load(json_file)
sectors = list(sectors_to_tickers.keys())

if 'last_sector_loaded' not in st.session_state:
    st.session_state['last_sector_loaded'] = '-'
    st.session_state['new_sector_load'] = False
    # this is setting some default ranges...
    st.session_state['current_nav_ranges'] = {
        'beta': (0.0, 5.0),
        'sharpe': (0.0, 5.0),
        'roi': (0.0, 5.0)        
    }


def execute(sector, beta, sharpe, roi):
    """
    This is the main data gathering for this app. It will call other functions
    to assemble a main dataframe which can be used in different ways.
    """

    # This is trying to detect if a new sector is loaded marking it in the session if
    # is is.
    if sector == st.session_state['last_sector_loaded']:
        st.session_state['new_sector_load'] = False
    else:
        st.session_state['last_sector_loaded'] = sector
        st.session_state['new_sector_load'] = True

    # unpacking the lows and highs into variables that can be used more easliy.
    beta_low, beta_high = beta
    sharpe_low, sharpe_high = sharpe
    roi_low, roi_high = roi

    main_df, mc_df = get_sector_data(sector)

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
    df_beta=pd.DataFrame(beta_s)
    df_beta.columns=['Beta']
    stats_df = pd.concat([df_roi, df_sharpe, df_std,df_beta], axis=1)
    

    if len(main_df.columns) > 0:
        st.line_chart(main_df)

        # EH: streamlit dataframe display
        st.subheader('Daily Closing Price')
        st.dataframe(main_df.dropna().style.highlight_max(axis=0))
        # EH: rates display on streamlit
        st.subheader('Rates')
        st.dataframe(stats_df.style.highlight_max(axis=0))

        #EH: Option to dispaly daily return chart
        if st.button('Daily Return Chart'): 
            st.line_chart(main_df.pct_change().dropna())
        else:
            st.write('Click the button to display Daily Return Chart')
        
        #EH: Option to dispaly Cumulative return chart
        if st.button('Cumulative Return Chart'):
            st.line_chart(((((main_df.pct_change().dropna()))+1).cumprod()-1).dropna())
        else:
            st.write('Click the button to display Cumulative Return Chart')
        
        #EH:  Option to display heatmap of correlation
        if st.button('Correlation Heatmap'):
            fig, ax = plt.subplots()
            sns.heatmap(main_df.corr(), ax=ax)
            st.write(fig)
        else:
            st.write('Click the button to display Correlation Heatmp.')

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
                    f'Please provide a weight percentage for {each}.',min_value=0,max_value=100)
                st.write(f'The current {each} weight percentage is ', number)
                weight_dict[each] = number
                confidence(each,'99%',main_df,df_std)
                confidence(each,'95%',main_df,df_std)        
        sum_weight_pct = sum(weight_dict.values())

        # EH:  error message for weight percent <>100.
        if sum_weight_pct != 100:
            st.error(
                    'Invalid weight percentage input.  The sum of weight percentage should be 100.')
        else:
            st.write('Thank you for the input!')



        if st.button('Run MC Return Simulation'):
            mc(mc_df, weight_dict)

            
        else: 
            st.write('Click button to see MC Return simulation based on your input.')
    else:
        st.write("No stocks matched the criteria selected")


def get_beta(main_df):
    daily_returns = main_df.pct_change().dropna()
    daily_returns.index = pd.to_datetime(daily_returns.index)
    # st.write(daily_returns)

    # Get SPY Dataframe
    spy_df = alpacaService.getLatestYearsData('SPY')
    
    spy_df = spy_df['SPY'].drop(columns=['open', 'high', 'low', 'volume'])
    spy_df_daily_returns = spy_df.pct_change().dropna()
    spy_var = spy_df_daily_returns['close'].var()
    spy_df_daily_returns = spy_df_daily_returns.tz_convert(None)
    spy_df_daily_returns.index = spy_df_daily_returns.index.date    

    # calculate cov and beta
    tickers_list = main_df.keys()
    beta_list = []

    beta_df = pd.Series()    
    for ticker in tickers_list:
        cov = daily_returns[ticker].cov(spy_df_daily_returns['close'])
        beta = (cov/spy_var)
        beta_df.loc[ticker] = beta

    return beta_df



def get_sector_data(sector):
    """
    This function is responsible for loading all of the stock data within a sector.
    """
    stocks_df = pd.DataFrame()
     
    stocks_to_load = sectors_to_tickers[sector]
    main_df = alpacaService.getLatestYearsData(stocks_to_load)
    
    #dropping unused columns
    main_df.drop(columns=['open','high','low','volume'], axis=1, level=1,inplace=True)

    #RA: Removing Timestamp and created copy of the main_df
    main_df.index = main_df.index.date   
    main_df.index.name = 'date'

    # need to keep an unflattend copy for the mc_simulator
    mc_df = main_df.copy(deep=True)

    # the y axis is multi-dementional, and this is flattening it.
    main_df.columns = [col[0] for col in main_df.columns.values]
    #closing_prices_df.columns = pd.MultiIndex.from_product([closing_prices_df.columns, ['closing']])

    return main_df, mc_df

    
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
    beta_df = get_beta(main_df)    

    # When a new sector is loaded, this is taking the highs and lows for the ranges
    # and setting them in the state. From there, it's trying to reload the page
    # with the proper values. 
    if st.session_state['new_sector_load'] == True:
        # Note - padding these values a little bit to acocunt for float percisssion inaccuracies.
        st.session_state['current_nav_ranges'] = {
            'beta': (beta_df.min()-.1, beta_df.max()+.1),            
            'sharpe': (sharpe_df.min()-.1, sharpe_df.max()+.1),
            'roi': (roi_df.min()-.1, roi_df.max()+.1)        
        }
        st.experimental_rerun()

    # All of these loops need to round the values to account for float percission inaccuracies.
    for ticker in roi_df.keys():
        roi = round(roi_df[ticker], 4)
        if ticker in main_df and (roi < round(roi_low,4) or roi > round(roi_high,4)):
            #st.write(">> roi dropping:", ticker, "::",roi_df[ticker], ":::", roi_low, roi_high)
            main_df.drop(columns=[ticker], axis=1, inplace=True)

    for ticker in sharpe_df.keys():
        if ticker in main_df and (sharpe_df[ticker] < sharpe_low or sharpe_df[ticker] > sharpe_high):
            #st.write(">> sharpe dropping:", ticker, "::",sharpe_df[ticker], ":::", sharpe_low, sharpe_high)
            main_df.drop(columns=[ticker], axis=1, inplace=True)

    for ticker in beta_df.index:
        if ticker not in beta_df:
            break

        beta = round(beta_df[ticker],4)
        if ticker in main_df and (beta < round(beta_low,4) or beta > round(beta_high,4)):
            #st.write(">> beta dropping:", ticker, "::",beta, ":::", round(beta_low,4), round(beta_high,4))
            main_df.drop(columns=[ticker], axis=1, inplace=True)

    return main_df


# EH:  caluclate confidence interval
ci_zscore_dict = {'99%': 2.576,
                  '95%': 1.96}

#EH:  define function to print confidence interval and its retuns
def confidence(stock, conf_pct,main_df,df_std):
    downside = main_df[stock].pct_change().dropna().mean() - ci_zscore_dict[conf_pct] * df_std.loc[stock][0]
    upside = main_df[stock].pct_change().dropna().mean() + ci_zscore_dict[conf_pct] * df_std.loc[stock][0]
    st.write(f"Using a {conf_pct} confidence interval, "
          f"the {stock} could trade down as much as {(downside * 100): .4f}%, "
          f"and up as much as {(upside * 100): .4f}%.")


# RA: Configure a Monte Carlo simulation to forecast five years cumulative returns
def mc(closing_prices_df, tickers_to_weights):

    rel_tickers = list(tickers_to_weights.keys())
    rel_weights = [x/100 for x in tickers_to_weights.values()]    

    # FIXME!!!! We really want to be using this, but this is only returning a 1 demension array, but
    # the MC simulator requires 2.
    rel_closing_prices_df = closing_prices_df.filter(items = rel_tickers)

    # HACK!!! we should be using the line above, but i have no idea how to convert that into the proper
    # format
    rel_closing_prices_df = alpacaService.getLatestYearsData(rel_tickers)    
    rel_closing_prices_df.drop(columns=['open','high','low','volume'], axis=1, level=1,inplace=True)
    rel_closing_prices_df.index = rel_closing_prices_df.index.date   
    rel_closing_prices_df.index.name = 'date'
    # End hack
    
    MC_fiveyear = MCSimulation(
        portfolio_data = rel_closing_prices_df,
        weights = rel_weights,
        num_simulation = 500,
        num_trading_days = 500
    )

    MC_fiveyear.calc_cumulative_return()
    # Plot simulation outcomes

    MC_sim_line_plot = MC_fiveyear.plot_simulation()
    MC_sim_line_plot.get_figure()
    # Save the plot for future use
    MC_sim_line_plot.get_figure().savefig("MC_fiveyear_sim_plot.png", bbox_inches="tight")
    img_path = Path("./MC_fiveyear_sim_plot.png")
    image = Image.open(img_path)
    st.image(image, caption='Monte Carlo Simulation')
    # Plot probability distribution and confidence intervals
    MC_sim_dist_plot = MC_fiveyear.plot_distribution()
    MC_sim_dist_plot.get_figure()
    # Save the plot for future use
    MC_sim_dist_plot.get_figure().savefig("MC_fiveyear_dist_plot.png", bbox_inches="tight", rot =90)
    img1_path = Path("./MC_fiveyear_dist_plot.png")
    image = Image.open(img1_path)
    st.image(image, caption='Monte Carlo Distribution')
    MC_summary_statistics = MC_fiveyear.summarize_cumulative_return()
    st.table(MC_summary_statistics)
    

def main():
    current_nav_ranges = st.session_state['current_nav_ranges']
    
    st.write()

    # Title
    st.title("Stock Selector App")

    #st.sidebar.title("Controls")
    st.sidebar.info( "Select the criteria that you want here.")
    sector = st.sidebar.selectbox("Sectors", sectors)
    beta = st.sidebar.slider('Beta Range', current_nav_ranges['beta'][0],
                             current_nav_ranges['beta'][1],
                             (current_nav_ranges['beta'][0],
                              current_nav_ranges['beta'][1]))
    sharpe = st.sidebar.slider('Sharpe Range', current_nav_ranges['sharpe'][0],
                               current_nav_ranges['sharpe'][1],
                               (current_nav_ranges['sharpe'][0],
                                current_nav_ranges['sharpe'][1]))
    roi = st.sidebar.slider('ROI Range',
                            current_nav_ranges['roi'][0],
                            current_nav_ranges['roi'][1],
                            (current_nav_ranges['roi'][0],
                             current_nav_ranges['roi'][1]))    
    execute(sector, beta, sharpe, roi)
    
main()  

