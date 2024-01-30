from PredictorsManagement.Predictors import Predictor
from keras import Sequential
from Utilities.FinUtils import *
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class RMResponse:
    '''a type of response that only risk managers give.
    :stoploss_selling_price: is the price at which the bot should place a stoploss order
    :risk: is True when money are at risk according to the risk_manager's conditions, False if it thinks everything's alright'''
    stoploss_price: float | None=None
    risk: bool | None=None

class RiskManager(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    def predict(self) -> RMResponse:
        pass

class FallPerceptionRiskManager(RiskManager):
    '''risk manager used for simulation of pseudo-dudes that only sell when their portfolio falls to a certain point so no limit orders. only returns bools'''
    def __init__(self, border_fall: float):
        self.border_fall = border_fall
    
    def predict(self, databit: Candles, avg_buy_price: float) -> RMResponse:
        stoploss_price = -1
        risk = False
        try:
            stoploss_price = avg_buy_price*(1-self.border_fall)
        except:
            print("couldn't calculate stoploss_price")
        try:
            risk = databit.Close.values[-1] <= avg_buy_price*(1-self.border_fall)
        except:
            print("couldn't calculate risk")
        return RMResponse(stoploss_price=stoploss_price, risk=risk)
    
class ATRRiskManager(RiskManager):
    '''returns the price at which a limit stop-loss order should be posted'''
    def __init__(self, border_atr: float=1):
        self.border_atr = border_atr
        
    def predict(self, databit: Candles, avg_buy_price: float) -> RMResponse:
        stoploss_price = -1
        risk = False
        try:
            stoploss_price = databit.Close.values[-1] - databit.ATR.values[-1] * self.border_atr
        except:
            print(f'ATRRiskManager cold not calculate stoposs_price.\nvalues provided: \nclose: {databit.Close.values[-1]} atr:{databit.ATR.values[-1]}')
        try:
            risk = (avg_buy_price <= stoploss_price) if avg_buy_price != 0 else False
        except:
            print(f"ATRRiskManager couldn't calculate risk.\nvalues provided:\n avg_buy_price:{avg_buy_price} and stoploss_price:{stoploss_price}")
        return RMResponse(stoploss_price=stoploss_price, risk=risk)
    
class AIRiskManager(RiskManager):
    def __init__(self, model: Sequential | Predictor):
        raise NotImplementedError