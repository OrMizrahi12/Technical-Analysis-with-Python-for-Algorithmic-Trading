
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")

class SOBacktester(): 
    ''' Class for the vectorized backtesting of SO-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    periods: int
        time window in days for rolling low/high
    D_mw: int
        time window in days for %D line
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
        sets one or two new SO parameters
        
    test_strategy:
        runs the backtest for the SO-based strategy
        
    plot_results:
        plots the performance of the strategy compared to buy and hold
        
    update_and_run:
        updates SO parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for the two SO parameters
    '''
    
    def __init__(self, symbol, periods, D_mw, start, end, tc):
        self.symbol = symbol    # instrument name
        self.periods = periods  # period size (for rolling min and rolling max)
        self.D_mw = D_mw        # period size for D% (rolling mean of K5)
        self.start = start      # start date 
        self.end = end          # end date
        self.tc = tc            # trading consts
        self.results = None     # store the results
        self.get_data()         # load the data 
        
    def __repr__(self):
        return "SOBacktester(symbol = {}, periods = {}, D_mw = {}, start = {}, end = {})".format(self.symbol, self.periods, self.D_mw, self.start, self.end)

    # Get the data 
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''

        raw = pd.read_csv("../Data/{}_ohlc.csv".format(self.symbol), parse_dates = [0], index_col = 0) # read
        raw = raw.dropna() # drop NaN
        raw = raw.loc[self.start:self.end] # extract the date range
        raw["returns"] = np.log(raw.Close / raw.Close.shift(1)) # compute log returns
        raw["roll_low"] = raw.Low.rolling(self.periods).min()   # rolling min (the lowest price of time period)
        raw["roll_high"] = raw.High.rolling(self.periods).max() # rolling max (the higher price of time period)
        raw["K"] = (raw.Close - raw.roll_low) / (raw.roll_high - raw.roll_low) * 100 # %k: the ratio between the most current price to the lowest price and the higher price
        raw["D"] = raw.K.rolling(self.D_mw).mean() # %D the SMA of %K
        self.data = raw

    # Set the SO parameters 
    def set_parameters(self, periods = None, D_mw = None):
        ''' Updates SO parameters and resp. time series.
        '''
        if periods is not None:
            self.periods = periods # re-set the period (size for rolling min & rolling max)
            self.data["roll_low"] = self.data.Low.rolling(self.periods).min() # compute the rolling min 
            self.data["roll_high"] = self.data.High.rolling(self.periods).max() # compute the rolling max
            self.data["K"] = (self.data.Close - self.data.roll_low) / (self.data.roll_high - self.data.roll_low) * 100 # re-compute %K
            self.data["D"] = self.data.K.rolling(self.D_mw).mean() # re-compute %D (SMA of %K)
        if D_mw is not None:
            self.D_mw = D_mw
            self.data["D"] = self.data.K.rolling(self.D_mw).mean()

    # Run rhw strategy        
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()

        # Potision strategy:
        # %K cross up %D ? signal to BUY (1)
        # %K cross down %D ? signal to SELL (-1)
        # Note: This is the basic strategy of Stocestic oselator, and there are another better strategies.
        data["position"] = np.where(data["K"] > data["D"], 1, -1)

        # Compute the return of the strategy
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # determine when a trade takes place (trade = change position)
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.tc
        
        # Cumelative return of buy and hold
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        # Cumelative return of SO strategy
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
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
            title = "{} | periods = {}, D_mw = {} | TC = {}".format(self.symbol, self.periods, self.D_mw, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

    # Callback function to minimize (by the optimizer)    
    def update_and_run(self, SO):
        ''' Updates SO parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        SO: tuple
            SO parameter tuple
        '''
        self.set_parameters(int(SO[0]), int(SO[1]))
        return -self.test_strategy()[0]
    
    # Optimizer function
    def optimize_parameters(self, periods_range, D_mw_range):
        ''' Finds global maximum given the SO parameter ranges.

        Parameters
        ==========
        periods_range, D_mw_range: tuple
            tuples of the form (start, end, step size)
        '''
        # Optimzier
        opt = brute(self.update_and_run, # function to minimize (find the best hyperparameters for SO) 
                    (periods_range, D_mw_range), # -> (periods range for rolling min/max , period range for %D SMA size)
                    finish=None)
        return opt, -self.update_and_run(opt) # return: the optimal hyperparameters, and its performance. 
    
    