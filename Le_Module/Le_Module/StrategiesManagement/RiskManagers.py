from PredictorsManagement.Predictors import Predictor
from keras import Sequential
from Utilities.FinUtils import *

class FallPerceptionRiskManager(Predictor):
    '''risk manager used for simulation of pseudo-dudes that only sell when their portfolio falls to a certain point so no limit orders. only returns bools'''
    def __init__(self, border_fall: float):
        self.border_fall = border_fall
    
    def predict(self, databit: Candles, avg_buy_price: float) -> bool:
        return (None, databit[-1].Close.values <= avg_buy_price*(1-self.border_fall))
    
class ATRRiskManager(Predictor):
    '''returns the price at which a limit stop-loss order should be posted'''
    def __init__(self, border_atr: float=1):
        self.border_atr = border_atr
        
    def predict(self, databit: Candles, avg_buy_price: float) -> tuple:
        try:
            stoploss_price = databit.Close.values[-1] - databit.ATR.values[-1] * self.border_atr
        except TypeError:
            print(f'databit.ATR is not of Indicator type. This may mean that no ATR was provided. What was provided tho is: \n{databit.ATR}')
        return (stoploss_price, avg_buy_price <= stoploss_price)
    
class AIRiskManager(Predictor):
    def __init__(self, model: Sequential | Predictor):
        raise NotImplementedError