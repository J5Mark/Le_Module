from Utilities.FinUtils import *
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray

class DecidingModule(ABC):
    '''this module decides on wether to buy or sell a security (B - True, S - False).'''
    def __init__(self, comission: float=0.05):
        ''':comission: comission rate that the broker has set for the security (percents)'''
        self.comission = comission

    @abstractmethod
    def _allow_trading(self, databit: Candles, trend: list | None = None) -> bool:
        return True

    @abstractmethod
    def decide(self, databit: Candles, trend: list | None = None, predictions: np.ndarray | None = None, risks: list | None = None) -> bool | None:
        if self._allow_trading(databit):
            pass
        else:
            return None
        
        
class AIDecicidingModuleTPR1(DecidingModule):
    def _allow_trading(self, databit: Candles, trend: list) -> bool:
        avg_trend = np.average([tr[0][1] for tr in trend])
        at = avg_trend*0.9 <= databit.Close.values[-1] and avg_trend > databit.Close.values[-1]
        return at
    
    def decide(self, databit: Candles, trend: list | None = None, predictions: ndarray | None = None, risks: list | None = None) -> bool | None:
        if any(risks[1]): return False
        if self._allow_trading(trend):
            return predictions[0][0][1] > databit[-1].Close.values*(1 + self.comission*0.01)
        else: return None
        
        
class SimpleAIDecidingModule(DecidingModule):   
    def _allow_trading(self, databit: Candles, trend: list | None = None) -> bool:
        return super()._allow_trading(databit, trend)
    
    def decide(self, databit: Candles, trend: list | None = None, predictions: ndarray | None = None, risks: list | None = None) -> bool | None:
        if any(risks): return False
        if self._allow_trading(trend):
            return predictions[0][0][1] > databit[-1].Close.values*(1 + self.comission*0.01)
        else: return None
