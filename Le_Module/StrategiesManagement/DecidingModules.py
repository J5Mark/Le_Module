from Utilities.FinUtils import *
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
import math as m
from PredictorsManagement.pm import *

from Utilities.FinUtils import Candles

class DecidingModule(ABC):
    '''this module decides on wether to buy or sell a security (B - True, S - False).'''
    def __init__(self, comission: float=0.05):
        ''':comission: comission rate that the broker has set for the security (percents)'''
        self.comission = comission

    def _allow_trading(self, databit: Candles, trend: list | None = None) -> bool:
        return True

    @abstractmethod
    def decide(self, databit: Candles, trend: list | None = None, predictions: np.ndarray | None = None, risks: list | None = None) -> bool | None:
        if self._allow_trading(databit):
            pass
        else:
            return None
        
        
class AIDecicidingModuleTPR1(DecidingModule):
    def _allow_trading(self, databit: Candles, trend: list[PredictorResponse]) -> bool:
        avg_trend = np.mean([tr.pred.biased for tr in trend])
        at = avg_trend*0.9 <= databit.Close.values[-1] and avg_trend > databit.Close.values[-1]
        return at
    
    def decide(self, databit: Candles, predictions: PredictorResponse, trend=None, risks: list | None = None) -> bool | None:
        if any(risks[1]): return False
        if self._allow_trading(trend):
            return predictions.pred.biased > databit[-1].Close.values*(1 + self.comission*0.01)
        else: return None
        
        
class SimpleAIDecidingModule(DecidingModule):   
    def decide(self, databit: Candles, trend: list | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        if any(risks): return False
        if self._allow_trading(trend):
            avg_pred = np.mean([p.pred.biased for p in predictions])
            return avg_pred > databit.Close.values[-1]*(1 + self.comission*0.01)
        else: return None

class MACrossing(DecidingModule):  
    def decide(self, databit: Candles, trend: list | None = None, predictions: ndarray | None = None, risks: list | None = None) -> bool | None:
        mid = m.floor(len(databit.MAs) / 2)
        a_ = 0
        b_ = 0
        for t in databit.MAs[:mid]:
            a_ += t
        for t in databit.MAs[-mid:]:
            b_ += t
        return (a_/mid) > (b_/(len(databit.MAs)-mid)) 
    
class TwoChosenMAsCrossing(DecidingModule):
    def __init__(self, fst: int=0, scd: int=-1):
        assert fst >= 0 
        assert scd >= 0 or scd == -1
        assert scd > fst or scd == -1
        self.least_ma_ind = fst
        self.greater_ma_ind = scd
        
    def decide(self, databit: Candles, trend: list | None = None, predictions: ndarray | None = None, risks: list | None = None) -> bool | None:
        assert len(databit.MAs) >= self.greater_ma_ind or self.greater_ma_ind == -1
        return databit.MAs[self.least_ma_ind] > databit.MAs[self.greater_ma_ind]
    
class BollingerBounce(DecidingModule):
    def __init__(self, bounce_up: float, bounce_down: float):
        self.bounce_up = bounce_up
        self.bounce_down = bounce_down
    
    def decide(self, databit: Candles, trend: list | None = None, predictions: ndarray | None = None, risks: list | None = None) -> bool | None:
        if databit.Close <= (1+self.bounce_up) * databit.BollingerBands[0] or databit.Low <= (1+self.bounce_up) * databit.BollingerBands[0]: 
            return True
        elif databit.Close >= (1+self.bounce_down) * databit.BollingerBands[2] or databit.High >= (1+self.bounce_down) * databit.BollingerBands[2]:
            return False
        return None