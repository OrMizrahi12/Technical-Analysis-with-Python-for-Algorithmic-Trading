
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")

class MACDBacktester(): 
    ''' Class for the vectorized backtesting of MACD-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    EMA_S: int
        time window in days for shorter EMA
    EMA_L: int
        time window in days for longer EMA
    signal_mw: int
        time window is days for MACD Signal 
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
    tc: float
        proportional transaction costs per trade
        
        
    Methods
    =======
    get_data:
        retrieves and prepares the data
        
    set_parameters:
        sets new MACD parameter(s)
        
    test_strategy:
        runs the backtest for the MACD-based strategy
        
    plot_results:
        plots the performance of the strategy compared to buy and hold
        
    update_and_run:
        updates MACD parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for the three MACD parameters
    '''
    
    def __init__(self, symbol, EMA_S, EMA_L, signal_mw, start, end, tc):
        self.symbol = symbol    # instrument
        self.EMA_S = EMA_S      # short EMA size (for Fast line)
        self.EMA_L = EMA_L      # long EMA size (for Fast line)
        self.signal_mw = signal_mw # EMA size (for Slow line)
        self.start = start         # start date of data
        self.end = end             # and date of data
        self.tc = tc               # trading costs per position
        self.results = None        # will store the strategy performance
        self.get_data()            # load the instrument data
        
    def __repr__(self):
        return "MACDBacktester(symbol = {}, MACD({}, {}, {}), start = {}, end = {})".format(self.symbol, self.EMA_S, self.EMA_L, self.signal_mw, self.start, self.end)
        
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        raw = pd.read_csv("../Data/forex_pairs.csv", parse_dates = ["Date"], index_col = "Date") # read the data
        raw = raw[self.symbol].to_frame().dropna() # extract the data of the specific instrument
        raw = raw.loc[self.start:self.end] # extrace the give range dates
        raw.rename(columns={self.symbol: "price"}, inplace=True) 
        
        raw["returns"] = np.log(raw / raw.shift(1)) # compute the log return
        
        # MACD (Fast line)
        raw["EMA_S"] = raw["price"].ewm(span = self.EMA_S, min_periods = self.EMA_S).mean() # Short EMA 
        raw["EMA_L"] = raw["price"].ewm(span = self.EMA_L, min_periods = self.EMA_L).mean() # Long EMA
        raw["MACD"] = raw.EMA_S - raw.EMA_L # compute the MACD (Fast line) -> the different between the EMA's 
        
        # MACD (Slow line)
        # this is the EMA of the MACD (Fast line). -> (avarage of avarage)
        raw["MACD_Signal"] = raw.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean() 
      
        self.data = raw
      
        
    def set_parameters(self, EMA_S = None, EMA_L = None, signal_mw = None):
        ''' Updates MACD parameters and resp. time series.
        '''
        # set the short EMA (for MACD Fast line)
        if EMA_S is not None:
            self.EMA_S = EMA_S
            self.data["EMA_S"] = self.data["price"].ewm(span = self.EMA_S, min_periods = self.EMA_S).mean() # compute short EMA
            self.data["MACD"] = self.data.EMA_S - self.data.EMA_L # recompute the MACD (Fast line)
            self.data["MACD_Signal"] = self.data.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean() # recompute the MACD (Slow line)
        
        # set the long EMA (for MACD Fast line)   
        if EMA_L is not None:
            self.EMA_L = EMA_L
            self.data["EMA_L"] = self.data["price"].ewm(span = self.EMA_L, min_periods = self.EMA_L).mean()  # compute long EMA
            self.data["MACD"] = self.data.EMA_S - self.data.EMA_L # recompute the MACD (Fast line)
            self.data["MACD_Signal"] = self.data.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean() # recompute the MACD (Slow line)

        # set the EMA (for MACD Slow line)   
        if signal_mw is not None:
            self.signal_mw = signal_mw 
            self.data["MACD_Signal"] = self.data.MACD.ewm(span = self.signal_mw, min_periods = self.signal_mw).mean() # recompute the MACD slow  
 
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()

        data["position"] = np.where(data["MACD"] > data["MACD_Signal"], 1, -1) # determine positions (MACD strategy based)
        data["strategy"] = data["position"].shift(1) * data["returns"] # compute the strategy return
        
        data.dropna(inplace=True)
        
        data["trades"] = data.position.diff().fillna(0).abs() # determine when a trade takes place
    
        data.strategy = data.strategy - data.trades * self.tc # subtract transaction costs from return when trade takes place
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp) # cumelative return of buy & hold strategy
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp) # cumelative return of MACD strategy
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | MACD ({}, {}, {}) | TC = {}".format(self.symbol, self.EMA_S, self.EMA_L, self.signal_mw, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

    # Callback Function for optimizer 
    def update_and_run(self, MACD):
        ''' Updates MACD parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        MACD: tuple
            MACD parameter tuple
        '''
        self.set_parameters(int(MACD[0]), int(MACD[1]), int(MACD[2])) # set MADC hyperparameters
        return -self.test_strategy()[0] # test the strategy, return the performance
    
    # Function to optimize and find the best hyper parameters! 
    def optimize_parameters(self, EMA_S_range, EMA_L_range, signal_mw_range):
        ''' Finds global maximum given the MACD parameter ranges.

        Parameters
        ==========
        EMA_S_range, EMA_L_range, signal_mw_range : tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(self.update_and_run, # function to minimize
                    (EMA_S_range, EMA_L_range, signal_mw_range), # MACD hyperparameter ranges 
                    finish=None) 
        return opt, -self.update_and_run(opt) # return the optimal hyperparameters, and the strategy absulute performance 
    
    