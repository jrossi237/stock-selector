import streamlit as st
import pandas as pd
import hvplot.pandas
import json
import param
from MCForecastTools import MCSimulation
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

alpacaService = AlpacaService(alpaca_key, alpaca_secret)

# Loading some base var used throughout the code.
sectors_to_tickers = {}
with open("sp_500_sectors_to_ticker.json") as json_file:
    sectors_to_tickers = json.load(json_file)
sectors = list(sectors_to_tickers.keys())

if 'last_sector_loaded' not in st.session_state:
    # This is initializing the state.
    
    st.session_state['last_sector_loaded'] = '-'
    st.session_state['new_sector_load'] = False
    # this is setting some default ranges...
    st.session_state['current_nav_ranges'] = {
        'beta': (0.0, 5.0),
        'sharpe': (0.0, 5.0),
        'roi': (0.0, 5.0)        
    }


def execute(sector, beta_ranges, sharpe_ranges, roi_ranges):
    """
    This is the main data gathering for this app. It will call other functions
    to assemble a main dataframe which can be used in different ways.

    Args:
        sector: the name of the sector to load.
        beta_ranges: the high and low beta range to filter upon
        sharpe_ranges: the high and low sharpe range to filter upon
        roi_ranges: the high and low roi to filter upon
    Return:
        None
    """

    # trying to detect if a new sector is loaded marking it in the session if is is.
    if sector == st.session_state['last_sector_loaded']:
        st.session_state['new_sector_load'] = False
    else:
        st.session_state['last_sector_loaded'] = sector
        st.session_state['new_sector_load'] = True

    main_df, mc_df = get_sector_data(sector)

    # EH:  call series data
    roi_s, sharpe_s, std_s = calculate_ratios(main_df)
    beta_s = get_beta(main_df)

    # Filtering the main df
    main_df = filter(main_df, beta_ranges, sharpe_ranges, roi_ranges, beta_s, sharpe_s, roi_s)    
    
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

        with st.expander("Daily Closing Price Reference"):
            st.dataframe(main_df.dropna().style.highlight_max(axis=0))
        # EH: rates display on streamlit
        
        with st.expander("ROI, Sharpe, STD, Beta Reference"):
            st.dataframe(stats_df.style.highlight_max(axis=0))

        #EH: Option to dispaly daily return chart
        if st.button('Daily Return Chart'): 
            st.line_chart(main_df.pct_change().dropna())
        else:
            st.write('')
        
        #EH: Option to dispaly Cumulative return chart
        if st.button('Cumulative Return Chart'):
            st.line_chart(((((main_df.pct_change().dropna()))+1).cumprod()-1).dropna())
        else:
            st.write('')
        
        #EH:  Option to display heatmap of correlation
        if st.button('Correlation Heatmap'):
            fig, ax = plt.subplots()
            sns.heatmap(main_df.corr(), ax=ax)
            st.write(fig)
        else:
            st.write('')

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
                with st.expander(f"{each} Return by Confidence Interval Reference"):
                    print_confidence(each,'99%',main_df,df_std)
                    print_confidence(each,'95%',main_df,df_std)        
        sum_weight_pct = sum(weight_dict.values())

        # EH:  error message for weight percent <>100.
        if len(weight_dict.keys()) > 0 and sum_weight_pct != 100:
            st.error(
                    'Invalid weight percentage input.  The sum of weight percentage should be 100.')
        else:
            st.write('Thank you for the input!')



        if st.button('Run MC Return Simulation'):
            with st.spinner('Exectuting Monte Carlo Simulator...'):
                run_monte_carlo(mc_df, weight_dict)
            
        else: 
            st.write('Click button to see MC Return simulation based on your input.')
    else:
        st.write("No stocks matched the criteria selected")


def get_beta(main_df):
    """
    Calculates the beta based on the main df that has been loaded.

    Args:
        main_df: The df to load everythign from.

    Returns:
        beta_s: the series which contains beta info

    """
    
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

    beta_s = pd.Series()    
    for ticker in tickers_list:
        cov = daily_returns[ticker].cov(spy_df_daily_returns['close'])
        beta = (cov/spy_var)
        beta_s.loc[ticker] = beta

    return beta_s

def get_sector_data(sector):
    """
    Loads the stock data for a given sector

    Args:
        sector: Name of the sector to load

    Returns:
        main_df: the main df which has been flattened.
        mc_df: the main df which has not been flatted. the mc similator needs an
            unflatted version of it.

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

    
def calculate_ratios(close_price_df):
    """
    Calculates the sharpe, roi, and std based on the main df that has been loaded.

    Args:
        close_price_df: The df that contains the close prices

    Returns:
        roi_s: series that contains all of the roi info
        sharpe_s: series that contains all of the sharpe info
        std_s: series that contains the standard deviations
    """
    
    # EH: daily rate
    daily_return_df = close_price_df.pct_change().dropna()
    # EH: cumulative return
    cumulative_return_df = ((daily_return_df+1).cumprod()-1).dropna()
    # EH:  get latest cumulative return value for filter purpose
    roi_s = cumulative_return_df.iloc[-1]
    #EH:  daily_return_mean_df
    daily_return_mean_df = daily_return_df.mean()
    #EH: std
    std_s = daily_return_df.std()
    #EH: ROI
    # annualized_return
    trade_days = 252
    annualized_return = daily_return_mean_df*trade_days
    # EH: annualized std
    annualized_std = std_s * (trade_days ** 1/2)
    # EH: sharpe ratio
    sharpe_s = annualized_return/annualized_std

    return roi_s, sharpe_s, std_s


def filter(main_df, beta_ranges, sharpe_ranges, roi_ranges, beta_s, sharpe_s, roi_s):
    """
    Filters the main_df based up the range selectors on the nav bar.

    Args:
        main_df: The main df that contains the close prices
        beta_ranges: a tuple that contains the high and low beta ranges
        sharpe_ranges: a tuple that contains the high and low sharpe ranges
        roi_ranges: a tuple that contains the high and low roi ranges
        beta_s: The beta series to use
        sharpe_s: The sharpe serioes to user
        roi_s: the roi series to use

    Returns:
        main_df: A filtered version of the main df.
    """
    
    # unpacking the lows and highs into variables that can be used more easliy.
    beta_low, beta_high = beta_ranges
    sharpe_low, sharpe_high = sharpe_ranges
    roi_low, roi_high = roi_ranges

    # When a new sector is loaded, this is taking the highs and lows for the ranges
    # and setting them in the state. From there, it's trying to reload the page
    # with the proper values. 
    if st.session_state['new_sector_load'] == True:
        # Note - padding these values a little bit to acocunt for float percisssion inaccuracies.
        st.session_state['current_nav_ranges'] = {
            'beta': (beta_s.min()-.1, beta_s.max()+.1),            
            'sharpe': (sharpe_s.min()-.1, sharpe_s.max()+.1),
            'roi': (roi_s.min()-.1, roi_s.max()+.1)        
        }
        st.experimental_rerun()

    # All of these loops need to round the values to account for float percission inaccuracies.
    for ticker in roi_s.keys():
        roi = round(roi_s[ticker], 4)
        if ticker in main_df and (roi < round(roi_low,4) or roi > round(roi_high,4)):
            #st.write(">> roi dropping:", ticker, "::",roi_df[ticker], ":::", roi_low, roi_high)
            main_df.drop(columns=[ticker], axis=1, inplace=True)

    for ticker in sharpe_s.keys():
        if ticker in main_df and (sharpe_s[ticker] < sharpe_low or sharpe_s[ticker] > sharpe_high):
            #st.write(">> sharpe dropping:", ticker, "::",sharpe_df[ticker], ":::", sharpe_low, sharpe_high)
            main_df.drop(columns=[ticker], axis=1, inplace=True)

    for ticker in beta_s.index:
        if ticker not in beta_s:
            break

        beta = round(beta_s[ticker],4)
        if ticker in main_df and (beta < round(beta_low,4) or beta > round(beta_high,4)):
            #st.write(">> beta dropping:", ticker, "::",beta, ":::", round(beta_low,4), round(beta_high,4))
            main_df.drop(columns=[ticker], axis=1, inplace=True)

    return main_df

#EH:  define function to print confidence interval and its retuns
def print_confidence(stock, conf_pct, main_df, df_std):
    """
    Calculates and prints out a message containing the confidence levels.

    Args:
        stock: stock ticker to use
        conf_pct: Confidence percent to use
        main_df: The df that contains the stock info
        df_std: the df which contains standard deviation info

    Returns:
        None
    """
    
    ci_zscore_dict = {'99%': 2.576,
                      '95%': 1.96}
    
    downside = main_df[stock].pct_change().dropna().mean() - ci_zscore_dict[conf_pct] * df_std.loc[stock][0]
    upside = main_df[stock].pct_change().dropna().mean() + ci_zscore_dict[conf_pct] * df_std.loc[stock][0]
    st.write(f"Using a {conf_pct} confidence interval, "
          f"the {stock} could trade down as much as {(downside * 100): .4f}%, "
          f"and up as much as {(upside * 100): .4f}%.")


# RA: Configure a Monte Carlo simulation to forecast five years cumulative returns
def run_monte_carlo(closing_prices_df, tickers_to_weights):
    """
    Run the monte carlo simulator on selected stocks

    Args:
        closing_prices_df: A dataframe of closing prices
        tickers_to_weights: a dict of tickers to their weights

    Returns:
        None
    """

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
    """
    Main function of this app. Sets up the side bar and then exectues the rest of the code.

    Returns:
        None
    """
    
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

