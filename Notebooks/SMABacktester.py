
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
plt.style.use("seaborn")


class SMABacktester():
    ''' Class for the vectorized backtesting of SMA-based trading strategies.

    Attributes
    ==========
    symbol: str
        ticker symbol with which to work with
    SMA_S: int
        time window in days for shorter SMA
    SMA_L: int
        time window in days for longer SMA
    start: str
        start date for data retrieval
    end: str
        end date for data retrieval
        
        
    Methods
    =======
    get_data:
        retrieves and prepares the data
        
    set_parameters:
        sets one or two new SMA parameters
        
    test_strategy:
        runs the backtest for the SMA-based strategy
        
    plot_results:
        plots the performance of the strategy compared to buy and hold
        
    update_and_run:
        updates SMA parameters and returns the negative absolute performance (for minimization algorithm)
        
    optimize_parameters:
        implements a brute force optimization for the two SMA parameters
    '''
    
    ## Constructure
    def __init__(self, symbol, SMA_S, SMA_L, start, end, ptc):
        self.symbol = symbol # instrument 
        self.SMA_S = SMA_S   # sort SMA 
        self.SMA_L = SMA_L   # long SMA
        self.start = start   # start date historical instrument data
        self.end = end       # end date historical instrument data
        self.results = None  # the cross SMA results (performance)
        self.get_data()      # load the data of the instrument
        self.ptc = ptc       # proportional trading cost por trade 

    ## String representation 
    def __repr__(self):
        return "SMABacktester(symbol = {}, SMA_S = {}, SMA_L = {}, start = {}, end = {}, ptc={})".format(self.symbol, 
                                                                                                         self.SMA_S,
                                                                                                         self.SMA_L,
                                                                                                         self.start,
                                                                                                         self.end,
                                                                                                         self.ptc)


    ## Get the data 
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        raw = pd.read_csv("../Data/forex_pairs.csv", parse_dates = ["Date"], index_col = "Date") # load from csv
        raw = raw[self.symbol].to_frame().dropna() # drop NaN
        raw = raw.loc[self.start:self.end] # extract only start to end dates
        raw.rename(columns={self.symbol: "price"}, inplace=True) # rename the column to 'price'
       
        raw["returns"] = np.log(raw / raw.shift(1)) # compute the log return 
        raw["SMA_S"] = raw["price"].rolling(self.SMA_S).mean() # compute sort SMA
        raw["SMA_L"] = raw["price"].rolling(self.SMA_L).mean() # compute long SMA 
       
        self.data = raw
    
    ## Re-set the hyperparameters strategy     
    def set_parameters(self, SMA_S = None, SMA_L = None): 
        ''' Updates SMA parameters and resp. time series.
        '''
        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean() # re-compute & re-set sort SMA 
        if SMA_L is not None:
            self.SMA_L = SMA_L
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean() # re-compute & re-set long SMA

    ## Test (run) the strategy        
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # determine when a trade takes place
        data["trades"] = data.position.diff().fillna(0).abs()
        # subtract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.ptc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        
        return round(perf, 6), round(outperf, 6)
        
    

    ## Plot the results
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | SMA_S = {} | SMA_L = {}".format(self.symbol, self.SMA_S, self.SMA_L) # title

            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8)) # plot

    ## Update the SMA's hyperparameters, then run the strategy (callback function for optimization function)   
    def update_and_run(self, SMA):
        ''' Updates SMA parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        SMA: tuple
            SMA parameter tuple
        '''
        self.set_parameters(int(SMA[0]), int(SMA[1])) # update sort/long SMAs
        return -self.test_strategy()[0] # return the absolute performance of the strategy
    
    ## Optimerzer function
    def optimize_parameters(self, SMA1_range, SMA2_range):
        ''' Finds global maximum given the SMA parameter ranges.

        Parameters
        ==========
        SMA1_range, SMA2_range: tuple
            tuples of the form (start, end, step size)
        '''
        
        # run the optimizer
        opt = brute(self.update_and_run, # callback function to minimize 
                   (SMA1_range, SMA2_range), # sort/long SMAs range
                   finish=None)

        return opt, -self.update_and_run(opt) # return: optimal SMAs hyperparameter, their results.   

