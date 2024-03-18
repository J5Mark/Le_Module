import random as rd
from PredictorsManagement.pm import PredictorResponse
from Utilities.DataClasses import Candles
import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod
from PredictorsManagement.pm import *

class QuantityController(ABC):
    '''this module decides on the amount of security lots to buy or sell.'''
    def __init__(self, comission: float=0.05, trend_importance: int=2):
        ''':comission: comission rate that the broker has set for the security (percents)
        :trend_importance: is how much to buy if overall trend is positive(if the q-controller even uses trend)'''
        self.comission = comission
        self.trend_importance = trend_importance
    @abstractmethod
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        pass

class AIQuantityControllerTPR1(QuantityController):
    '''amount of security lots to be taken increases the further the predictions assume the price to ascend
    if the bot is going to sell, this thig decides to sell it all'''

    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        if any(risks): return -1
        if direction:
            N = 1
            if predictions[0].pred.biased > databit.Close.values[-1]*(1 + self.comission*0.01):
                for p in range(1, len(predictions)):
                    if predictions[p].pred.biased > predictions[p-1].pred.biased:
                        N = p+1
        else: N = -1 #-1 meaning all
        return N
    
class AIQuantityControllerTR1(QuantityController):
    '''If current price is less than overall where-its-heading, buy trend_importance'''        
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        if any(risks): return -1
        if direction:
            N = 1
            if np.mean([t.pred.biased for t in trend]) > databit.Close.values[-1] * (1 + self.comission*0.01):
                N = self.trend_importance
        else: N = -1
        return N
    
class AIQuantityControllerTPR2(QuantityController):
    '''amount of security lots to be taken increases the further the predictions assume the price to ascend
    if the bot is going to sell, this thig decides to sell it all. 
    If the longer term trend is greater than the greatest prediction, buy trend_importance of security'''
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        if any(risks): return -1
        if direction:
            N = 1
            if predictions[0].pred.biased > databit.Close.values[-1]*(1 + self.comission*0.01):
                for p in range(1, len(predictions)):
                    if predictions[p].pred.biased > predictions[p-1].pred.biased:
                        N = p+1
                        
            if max([t.pred.biased for t in predictions]) < np.mean([b.pred.biased for b in trend]):
                N = self.trend_importance
        else: N = -1 #-1 meaning all
        return N
    
class AIQuantityControllerBPR_avg(QuantityController):
    '''average bias is taken into account. If "the worst case scenario" is better than the current price buy 2, 
    if only the middle prediction is greater than the current price, buy 1, else - don`t buy'''
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        if any(risks): return -1
        if direction:
            if predictions[0].pred.avg_bias_adjusted[0] > databit.Close.values[-1] * (1 + self.comission*0.01):
                N = 2
            elif predictions[0].pred.biased > databit.Close.values[-1] * (1 + self.comission*0.01):
                N = 1
            else:
                N = 0
        else: N = -1
        return N
    
class AIQuantityControllerBPR_med(QuantityController):
    '''median bias is taken into account. If "the worst case scenario" is better than the current price buy 2, 
    if only the middle prediction is greater than the current price, buy 1, else - don`t buy'''
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        if any(risks): return -1
        if direction:
            if predictions[0].pred.med_bias_adjusted[0] > databit[-1].Close.values[-1] * (1 + self.comission*0.01):
                N = 2
            elif predictions[0].pred.biased > databit[-1].Close.values[-1] * (1 + self.comission*0.01):
                N = 1
            else:
                N = 0
        else: N = -1
        return N
    
class AIQuantityControllerR1(QuantityController):
    '''if the direction wants to buy, buy one, otherwise sell all'''
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        if any(risks): return -1
        if direction:
            N = 1
        else: N = -1
        return N

class RandomQuantityController(QuantityController):
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        return rd.randint(1, 10) if direction else -1
    
class RandomSellingBlocked(QuantityController):
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        return rd.randint(1, 10) if direction else 0
    
class RandomQithHodlSellingBlocked(QuantityController):
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        return rd.choice([0,0,0,0,*list(range(1,11))]) if direction else 0
    
class MyComplexQuantity(QuantityController):
    def __init__(self, bounce_up: float=0, bounce_down: float=0, greater_ma_ind: int=-1, least_ma_ind: int=0, trend_coef: float=1):
        self.trend_coef = trend_coef
        self.bounce_up = bounce_up
        self.bounce_down = bounce_down
        self.greater_ma_ind = greater_ma_ind
        self.least_ma_ind = least_ma_ind
        
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        if direction == True:
            return sum([databit.Close <=  databit.BollingerBands[0] * (1+self.bounce_up), 
                    databit.Low <= databit.BollingerBands[0] * (1+self.bounce_up),     # bollinger_bounce thing here
                    databit.MAs[self.least_ma_ind] > databit.MAs[self.greater_ma_ind], # crossing mas thig
                    databit.MAs[self.least_ma_ind] > databit.Close,  # least span ma higher than price
                    trend[0].pred.biased > databit.Close]) #trend taken in account
        elif direction == None:
            return 0
        elif direction == False:
            return -1
        
class MyComplexQuantityNoAI(QuantityController):
    def __init__(self, bounce_up: float=0, bounce_down: float=0, greater_ma_ind: int=-1, least_ma_ind: int=0):
        self.bounce_up = bounce_up
        self.bounce_down = bounce_down
        self.greater_ma_ind = greater_ma_ind
        self.least_ma_ind = least_ma_ind
        
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None) -> int:
        if direction == True:
            return sum([databit.Close <=  databit.BollingerBands[0] * (1+self.bounce_up),
                    databit.Low <= databit.BollingerBands[0] * (1+self.bounce_up),     # bollinger_bounce thing here
                    databit.MAs[self.least_ma_ind] > databit.MAs[self.greater_ma_ind], # crossing mas thig
                    databit.MAs[self.least_ma_ind] > databit.Close]) #least ma crossing closing price
        elif direction == None:
            return 0
        elif direction == False:
            return -1
        
class FixedQuantity(QuantityController):
    def __init__(self, quantity_to_buy: int=1, quantity_to_sell: int=-1):
        self.buy = quantity_to_buy
        self.sell = quantity_to_sell
        
    def decide_quantity(self, databit: Candles, direction: bool, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list[bool] | None = None, budget: float | None=None):
        return self.buy if direction else self.sell