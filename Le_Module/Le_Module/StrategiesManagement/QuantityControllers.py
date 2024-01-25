import random as rd
from Utilities.FinUtils import Candles
import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod

class QuantityController(ABC):
    '''this module decides on the amount of security lots to buy or sell.'''
    def __init__(self, comission: float=0.05, trend_importance: int=2):
        ''':comission: comission rate that the broker has set for the security (percents)
        :trend_importance: is how much to buy if overall trend is positive(if the q-controller even uses trend)'''
        self.comission = comission
        self.trend_importance = trend_importance
    @abstractmethod
    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: np.ndarray | None = None, risks: list | None = None) -> int:
        pass

class AIQuantityControllerTPR1(QuantityController):
    '''amount of security lots to be taken increases the further the predictions assume the price to ascend
    if the bot is going to sell, this thig decides to sell it all'''

    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: np.ndarray | None = None, risks: list | None = None) -> int:
        if any(risks): return -1
        if direction:
            N = 1
            if predictions[0][0][1] > databit.Close.values[-1]*(1 + self.comission*0.01):
                for p in range(1, len(predictions)):
                    if predictions[p][0][1] > predictions[p-1][0][1]:
                        N = p+1
        else: N = -1 #-1 meaning all
        return N
    
class AIQuantityControllerTR1(QuantityController):
    '''If current price is less than overall where-its-heading, buy trend_importance'''        
    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: np.ndarray | None = None, risks: list | None = None) -> int:
        if any(risks): return -1
        if direction:
            N = 1
            if np.mean(trend) > databit[-1].Close.values * (1 + self.comission*0.01):
                N = self.trend_importance
        else: N = -1
        return N
    
class AIQuantityControllerTPR2(QuantityController):
    '''amount of security lots to be taken increases the further the predictions assume the price to ascend
    if the bot is going to sell, this thig decides to sell it all. 
    If the longer term trend is greater than the greatest prediction, buy trend_importance of security'''
    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: np.ndarray | None = None, risks: list | None = None) -> int:
        if any(risks): return -1
        if direction:
            N = 1
            if predictions[0][0][1] > databit.Close.values[-1]*(1 + self.comission*0.01):
                for p in range(1, len(predictions)):
                    if predictions[p][0][1] > predictions[p-1][0][1]:
                        N = p+1
                        
            if max(predictions[0][0]) < np.mean(trend):
                N = self.trend_importance
        else: N = -1 #-1 meaning all
        return N
    
class AIQuantityControllerBPR_avg(QuantityController):
    '''average bias is taken into account. If "the worst case scenario" is better than the current price buy 2, 
    if only the middle prediction is greater than the current price, buy 1, else - don`t buy'''
    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: ndarray | None = None, risks: list | None = None) -> int:
        if any(risks): return -1
        if predictions[0][0][0] > databit.Close.values[-1] * (1 + self.comission*0.01):
            N = 2
        elif predictions[0][0][1] > databit.Close.values[-1] * (1 + self.comission*0.01):
            N = 1
        else:
            N = -1
        return N
    
class AIQuantityControllerBPR_med(QuantityController):
    '''median bias is taken into account. If "the worst case scenario" is better than the current price buy 2, 
    if only the middle prediction is greater than the current price, buy 1, else - don`t buy'''
    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: ndarray | None = None, risks: list | None = None) -> int:
        if any(risks): return -1
        if predictions[0][1][0] > databit[-1].Close.values[-1] * (1 + self.comission*0.01):
            N = 2
        elif predictions[0][1][1] > databit[-1].Close.values[-1] * (1 + self.comission*0.01):
            N = 1
        else:
            N = 0
        return N
    
class AIQuantityControllerR1(QuantityController):
    '''if the direction wants to buy, buy one, otherwise sell all'''
    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: np.ndarray | None = None, risks: list | None = None) -> int:
        if any(risks): return -1
        if direction:
            N = 1
        else: N = -1
        return N

class RandomQuantityController(QuantityController):
    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: ndarray | None = None, risks: list | None = None) -> int:
        return rd.randint(1, 10) if direction else -1
    
class RandomSellingBlocked(QuantityController):
    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: ndarray | None = None, risks: list | None = None) -> int:
        return rd.randint(1, 10) if direction else 0
    
class RandomQithHodlSellingBlocked(QuantityController):
    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: ndarray | None = None, risks: list | None = None) -> int:
        return rd.choice([0,0,0,0,*list(range(1,11))]) if direction else 0