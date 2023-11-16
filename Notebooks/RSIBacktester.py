
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")

class RSIBacktester(): 
    ''' Class for the vectorized backtesting of RSI-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    periods: int
        time window in days to calculate moving average UP & DOWN 
    rsi_upper: int
        upper rsi band indicating overbought instrument
    rsi_lower: int
        lower rsi band indicating oversold instrument
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
        sets new RSI parameter(s)
        
    test_strategy:
        runs the backtest for the RSI-based strategy
        
    plot_results:
        plots the performance of the strategy compared to buy and hold
        
    update_and_run:
        updates RSI parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for the three RSI parameters
    '''
    
    def __init__(self, symbol, periods, rsi_upper, rsi_lower, start, end, tc):
        self.symbol = symbol      # the instrument
        self.periods = periods    # time window in days to calculate moving average UP & DOWN (gain/loss rolling avarage)
        self.rsi_upper = rsi_upper # RSI upper bound (typically 70%)
        self.rsi_lower = rsi_lower # RSI lower bound (typically 30%)
        self.start = start         # start date data 
        self.end = end             # end date data
        self.tc = tc               # trading costs per trade
        self.results = None        # data frame for store the strategy performance
        self.get_data()            # load the data 
        
    def __repr__(self):
        return "RSIBacktester(symbol = {}, RSI({}, {}, {}), start = {}, end = {})".format(self.symbol, self.periods, self.rsi_upper, self.rsi_lower, self.start, self.end)

    # Get And Prepare The Data    
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        raw = pd.read_csv("../Data/forex_pairs.csv", parse_dates = ["Date"], index_col = "Date") # read the data
        raw = raw[self.symbol].to_frame().dropna() # extracr only the instrument data
        raw = raw.loc[self.start:self.end] # extract the given range
        raw.rename(columns={self.symbol: "price"}, inplace=True) # rename the column

        raw["returns"] = np.log(raw / raw.shift(1)) # compute lof return 
        
           
        raw["U"] = np.where(raw.price.diff() > 0, raw.price.diff(), 0) # Up (save the dates where the return positiv) - gain/profit days. else - set 0.         
        raw["D"] = np.where(raw.price.diff() < 0, -raw.price.diff(), 0) # Down (save the dates where the return negative) - loss days. else - set 0
        
        raw["MA_U"] = raw.U.rolling(self.periods).mean() # take the rolling mean of gains (profit) of the last `periods` time 
        raw["MA_D"] = raw.D.rolling(self.periods).mean() # take the roilling mean of loss of the last `periods` time
        raw["RSI"] = raw.MA_U / (raw.MA_U + raw.MA_D) * 100 # then, compute RSI! (the ratio between up and down, givs a range of: 0%-100%). 
        
        self.data = raw 

    
    # Set the RSI hyper parameters (if you want to change...)!
    def set_parameters(self, periods = None, rsi_upper = None, rsi_lower = None):
        ''' Updates RSI parameters and resp. time series.
        '''
        # update the periods (size of rolling avarage (e.g from 20 to 25))
        if periods is not None:
            self.periods = periods     
            self.data["MA_U"] = self.data.U.rolling(self.periods).mean() # moving avarage UP (gains)
            self.data["MA_D"] = self.data.D.rolling(self.periods).mean() # moving avarage DOWN (loss)
            self.data["RSI"] = self.data.MA_U / (self.data.MA_U + self.data.MA_D) * 100 # re-compute the RSI 

        # re set the upper bound    
        if rsi_upper is not None:
            self.rsi_upper = rsi_upper

        # re set the lower bound    
        if rsi_lower is not None:
            self.rsi_lower = rsi_lower

    # Test (rin the strategy!)        
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()

        data["position"] = np.where(data.RSI > self.rsi_upper, -1, np.nan) # RSI > upper bound? SELL (-1).
        data["position"] = np.where(data.RSI < self.rsi_lower, 1, data.position)  # RSI < lower bound? BUY (1).
        data.position = data.position.fillna(0) # otherwise, go nutral (0). 
        
        data["strategy"] = data["position"].shift(1) * data["returns"] # compute the return of the strategy
        data.dropna(inplace=True)
        
        data["trades"] = data.position.diff().fillna(0).abs() # determine when a trade takes place 
        
        data.strategy = data.strategy - data.trades * self.tc # subtract transaction costs from return when trade takes place
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp) # cumelative return of buy and hold 
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp) # cumelative return of RSI
        
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        
        return round(perf, 6), round(outperf, 6)
    

    # Plot the results (cumelatuve return)!
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | RSI ({}, {}, {}) | TC = {}".format(self.symbol, self.periods, self.rsi_upper, self.rsi_lower, self.tc)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

    # Callback function for to mizimize (by the optimizer)    
    def update_and_run(self, RSI):
        ''' Updates RSI parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        RSI: tuple
            RSI parameter tuple
        '''
        self.set_parameters(int(RSI[0]), int(RSI[1]), int(RSI[2])) # reset parameters
        return -self.test_strategy()[0] # run & return strategy absulute performence
    
    # The optimizer! 
    def optimize_parameters(self, periods_range, rsi_upper_range, rsi_lower_range):
        ''' Finds global maximum given the RSI parameter ranges.

        Parameters
        ==========
        periods_range, rsi_upper_range, rsi_lower_range : tuple
            tuples of the form (start, end, step size)
        '''
        opt = brute(
            self.update_and_run, # function to minimize! (find the best RSI hyperparameters)
                    (
                     periods_range,   # range of periods (moving avarage size) 
                     rsi_upper_range, # range for upper bound
                     rsi_lower_range  # range for lower bound
                    ),
            finish=None
        )
        
        return opt, -self.update_and_run(opt) # return the optimal hyperparameter and its performance
    
    