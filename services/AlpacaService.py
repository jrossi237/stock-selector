import pandas as pd
import alpaca_trade_api as tradeapi
import streamlit as st

class AlpacaService:
    """
    Basic service to get alpaca data.
    """
    
    cache_key = 'alpaca_service_cache'
    
    def __init__(self, alpaca_key, alpaca_secret):
        self.alpaca = tradeapi.REST(alpaca_key, alpaca_secret, api_version='v2')
        self.alpaca_trading = tradeapi.REST(alpaca_key, alpaca_secret, base_url="https://paper-api.alpaca.markets", api_version='v2')

        if  self.cache_key not in st.session_state:
            st.session_state[self.cache_key] = {}
           
    def buystock (self, stock_investment):
        for key,value in stock_investment.items():
            self.alpaca_trading.submit_order(
            symbol=key,
            notional = value,
            side='buy',
            type='market',
            time_in_force='day'
        )
        
    
    def getLatestYearsData(self, tickers):
        """
        Gets the latest year of stock data for tickers that are passed into it.

        Args:
            tickers: string or array of tickers to get.
        """
        start = (pd.Timestamp.now() - pd.Timedelta(days=365)).isoformat()
        end = pd.Timestamp.now().isoformat()

        key = None

        if isinstance(tickers, list):
            key = start.split("T")[0] + "-" + end.split("T")[0] + "-".join(tickers)
        else:
            key = start.split("T")[0] + end.split("T")[0] + tickers
        #st.write(key)        

        if key in st.session_state[self.cache_key]:
            #st.write("hit cache!!!", type(st.session_state[self.cache_key][key]))
            return st.session_state[self.cache_key][key].copy(deep=True)
        
        df = self.alpaca.get_barset(tickers, start=start, end=end, timeframe='1D', limit=252).df
        st.session_state[self.cache_key][key] = df
        
        return df.copy(deep=True)

        
        
