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
    
    def predict(self, databit: Candles, budget: float | None=None, avg_buy_price: float | None=None, last_buy_price: float | None=None, first_buy_price: float | None=None, predictions: list[PredictorResponse] | None=None) -> RMResponse:
        try:
            stoploss_price = avg_buy_price*(1-self.border_fall)
        except:
            print(f"{self.__class__.__name__} couldn't calculate stoploss_price")
        try:
            risk = databit.Close.values[-1] <= avg_buy_price*(1-self.border_fall)
        except:
            print(f"{self.__class__.__name__} couldn't calculate risk")
        return RMResponse(stoploss_price=stoploss_price, risk=risk)
    
class SketchyATRRiskManager(RiskManager): #very strange, almost sketchy
    '''returns the price at which a limit stop-loss order should be posted'''
    def __init__(self, border_atr: float=0.75):
        self.border_atr = border_atr
        
    def predict(self, databit: Candles, avg_buy_price: float | None=None, last_buy_price: float | None=None, first_buy_price: float | None=None, predictions: list[PredictorResponse] | None=None) -> RMResponse:
        stoploss_price = -1
        risk = False
        try:
            stoploss_price = databit.Close.values[-1] - databit.ATR.values[-1] * self.border_atr
        except:
            print(f'{self.__class__.__name__} could not calculate stoposs_price.\nvalues provided: \nclose: {databit.Close.values[-1]} atr:{databit.ATR.values[-1]}')
        try:
            risk = (avg_buy_price <= stoploss_price) if avg_buy_price != 0 else False
        except:
            print(f"{self.__class__.__name__} couldn't calculate risk.\nvalues provided:\n avg_buy_price:{avg_buy_price} and stoploss_price:{stoploss_price}")
        return RMResponse(stoploss_price=stoploss_price, risk=risk)
    
class AvgPriceATRRiskManager(RiskManager):
    def __init__(self, border_atr: float=0.75):
        self.border_atr = border_atr
        self.highest = 0
        
    def predict(self, databit: Candles, budget: float | None = None, avg_buy_price: float | None=None, last_buy_price: float | None=None, first_buy_price: float | None=None, predictions: list[PredictorResponse] | None=None) -> RMResponse:
        try:
            if self.highest < avg_buy_price: 
                self.highest = avg_buy_price
            stoploss_price = self.highest - databit.ATR.values[-1]*self.border_atr
        except:
            print(f'{self.__class__.__name__} cold not calculate stoposs_price.\nvalues provided: \nclose: {databit.Close.values[-1]} atr:{databit.ATR.values[-1]}')
        try:
            risk = databit.Close.values[-1] <= stoploss_price
            if risk:
                self.highest = 0
        except:
            print(f"{self.__class__.__name__} couldn't calculate risk.\nvalues provided:\n avg_buy_price:{avg_buy_price} and stoploss_price:{stoploss_price}")
        return RMResponse(stoploss_price=stoploss_price, risk=risk)
    
class PeakPriceATRRiskManager(RiskManager):
    def __init__(self, border_atr: float=0.75):
        self.border_atr = border_atr
        self.peak = 0
        self.sl = 0

    def predict(self, databit: Candles, avg_buy_price: float | None=None, last_buy_price: float | None=None, first_buy_price: float | None=None, predictions: list[PredictorResponse] | None=None) -> RMResponse:
        self.peak = databit.Close.values[-1] if databit.Close.values[-1] > self.peak else self.peak
        try:
            stoploss_price = max(self.peak - databit.ATR.values[-1]*self.border_atr, self.sl)
        except:
            print(f'{self.__class__.__name__} cold not calculate stoposs_price.\nvalues provided: \nclose: {databit.Close.values[-1]} atr:{databit.ATR.values[-1]}')
        try:
            risk = databit.Close.values[-1] <= stoploss_price
            if risk:
                self.sl = 0
        except:
            print(f"{self.__class__.__name__} couldn't calculate risk.\nvalues provided:\n avg_buy_price:{avg_buy_price} and stoploss_price:{stoploss_price}")
        
        return RMResponse(stoploss_price=stoploss_price, risk=risk)
        
class LastPriceATRRiskManager(RiskManager):
    def __init__(self, border_atr: float=0.75):
        self.border_atr = border_atr
        
    def predict(self, databit: Candles, avg_buy_price: float | None=None, last_buy_price: float | None=None, first_buy_price: float | None=None, predictions: list[PredictorResponse] | None=None) -> RMResponse:
        try:
            stoploss_price = last_buy_price - databit.ATR.values[-1]*self.border_atr
        except:
            print(f'{self.__class__.__name__} cold not calculate stoposs_price.\nvalues provided: \nclose: {databit.Close.values[-1]} atr:{databit.ATR.values[-1]}')
        try:
            risk = databit.Close.values[-1] <= stoploss_price
        except:
            print(f"{self.__class__.__name__} couldn't calculate risk.\nvalues provided:\n avg_buy_price:{avg_buy_price} and stoploss_price:{stoploss_price}")
        return RMResponse(stoploss_price=stoploss_price, risk=risk)
    
class FirstPriceATRRiskManager(RiskManager):
    def __init__(self, border_atr: float=0.75):
        self.border_atr = border_atr
        
    def predict(self, databit: Candles, avg_buy_price: float | None=None, last_buy_price: float | None=None, first_buy_price: float | None=None, predictions: list[PredictorResponse] | None=None) -> RMResponse:
        try:
            stoploss_price = first_buy_price - databit.ATR.values[-1]*self.border_atr
        except:
            print(f'{self.__class__.__name__} cold not calculate stoposs_price.\nvalues provided: \nclose: {databit.Close.values[-1]} atr:{databit.ATR.values[-1]}')
        try:
            risk = databit.Close.values[-1] <= stoploss_price
        except:
            print(f"{self.__class__.__name__} couldn't calculate risk.\nvalues provided:\n avg_buy_price:{avg_buy_price} and stoploss_price:{stoploss_price}")
        return RMResponse(stoploss_price=stoploss_price, risk=risk)
    
class AIRiskManager(RiskManager):
    def __init__(self, model: Sequential | Predictor):
        raise NotImplementedError