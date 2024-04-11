from PredictorsManagement.pm import PredictorResponse
from Utilities.FinUtils import *
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
import math as m
from PredictorsManagement.pm import *

from Utilities.FinUtils import Candles, PredictorResponse

class DecidingModule(ABC):
    '''this module decides on wether to buy or sell a security (B - True, S - False).
    parameter_controller can be used to influence DecidingModule`s parameters based on different market conditions'''
    
    class parameter_controller(ABC):
        def __init__(self, preparing_data: ParameterControllerData | None=None):
            self.prepare_controller()
        
        def prepare_controller(self):
            pass
        
        def control_params(self):
            pass
    
    def __init__(self, comission: float=0.05, preparing_data: ParameterControllerData | None=None):
        ''':comission: comission rate that the broker has set for the security (percents)'''
        self.comission = comission
        if any([preparing_data]):
            self.paramcontroller = self.parameter_controller()

    def _allow_trading(self, databit: Candles, trend: list | None = None) -> bool:
        return True

    @abstractmethod
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        if self._allow_trading(databit):
            pass
        else:
            return None
        
        
class AIDecicidingModuleTPR1(DecidingModule):
    def _allow_trading(self, databit: Candles, trend: list[PredictorResponse]) -> bool:
        avg_trend = np.mean([tr.pred.biased for tr in trend])
        at = avg_trend*0.9 <= databit.Close.values[-1] and avg_trend > databit.Close.values[-1]
        return at
    
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        if any(risks[1]): return False
        if self._allow_trading(trend):
            return predictions[0].pred.biased > databit[-1].Close.values*(1 + self.comission*0.01)
        else: return None
        
class AIDecidingModuleBPR_avg(DecidingModule):
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        if not any(risks):
            if any([predictions[0].pred.avg_bias_adjusted[0] >= databit.Close.values[-1],
                    predictions[0].pred.biased > databit.Close.values[-1]]):
                return True
            elif predictions[0].pred.avg_bias_adjusted[1] > databit.Close.values[-1]:
                return None
            else:
                return False
        else:
            return False
        
class SimpleAIDecidingModule(DecidingModule):   
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        if any(risks): return False
        if self._allow_trading(trend):
            avg_pred = np.mean([p.pred.biased for p in predictions])
            return avg_pred > databit.Close.values[-1]*(1 + self.comission*0.01)
        else: return None

class MACrossing(DecidingModule):  
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool:
        mid = m.floor(len(databit.MAs) / 2)
        a_ = 0
        b_ = 0
        for t in databit.MAs[:mid]:
            a_ += t.values[-1]
        for t in databit.MAs[-mid:]:
            b_ += t.values[-1]
        return (a_/mid) > (b_/(len(databit.MAs)-mid)) 
    
class MACloseCrossing(DecidingModule):
    def __init__(self, ma_ind: int=0):
        self.ma_ind = ma_ind
        
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None: 
        assert self.ma_ind <= len(databit.MAs) - 1
        return databit.MAs[self.ma_ind] > databit.Close
    
class f(DecidingModule): 
    '''lt f be the class for experimental strategies'''
    def __init__(self, strictness: float=0.01, k: float=1*10**(-7)):
        self.strictness = strictness
        self.k = k
        
    def _allow_trading(self, databit: Candles, trend: list | None = None) -> bool:
        c1 = detect_last_collision(databit.Close, databit.BollingerBands[0], strictness=self.strictness)
        c2 = detect_last_collision(databit.Close, databit.BollingerBands[2], strictness=self.strictness)
        
        return c1.ago < c2.ago if c2.ago is not None and c1.ago is not None else c1.ago is not None   #should also add some trendviewing for sure
        
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None: 
        if self._allow_trading(databit, trend):
            current_ma_speed = databit.MAs[0].values[-1] - databit.MAs[0].values[-2]
            return current_ma_speed >= self.k*databit.MAs[0].values[-1] or detect_last_collision(databit.Close, databit.BollingerBands[0]).ago <= 1
        else:
            return None
                
    
class EMACloseCrossing(DecidingModule):
    def __init__(self, ema_ind: int=0):
        self.ema_ind = ema_ind
    
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        assert self.ema_ind <= len(databit.EMAs) - 1
        return databit.EMAs[self.ema_ind] > databit.Close
    
class MAGrowingSpeed0(DecidingModule):
    def __init__(self, k: float=1*10**(-7)):
        self.k = k
    
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        assert len(databit.MAs[0]) >= 2
        current_ma_speed = databit.MAs[0].values[-1] - databit.MAs[0].values[-2]
        
        return current_ma_speed >= self.k*databit.MAs[0].values[-1]  #####works suspiciously well, gotta test you on some other things    

class EMAGrowingSpeed0(DecidingModule):
    def __init__(self, k: float=1*10**(-7)):
        self.k = k
    
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        assert len(databit.EMAs[0]) >= 2
        current_ma_speed = databit.EMAs[0].values[-1] - databit.EMAs[0].values[-2]
        
        return current_ma_speed >= self.k*databit.EMAs[0].values[-1]

class MAGrowingSpeedIgnores(DecidingModule):
    def __init__(self, k: float=1*10**(-7), vicinity: float=1*10**(-2)):
        self.k = k
        self.vicinity = vicinity
        
    def _allow_trading(self, databit: Candles, trend: list | None = None) -> bool:
        return not any([databit.Low.values[-1] >= databit.MAs[-1].values[-1]*(1-self.vicinity),
                        databit.Low.values[-1] <= databit.MAs[-1].values[-1]*(1+self.vicinity),
                        databit.High.values[-1] >= databit.MAs[-1].values[-1]*(1-self.vicinity),
                        databit.High.values[-1] <= databit.MAs[-1].values[-1]*(1+self.vicinity)]) #also uses a risk perception+atr rm
        
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        assert len(databit.MAs[0]) < 3
        current_ma_speed = databit.MAs[0].values[-1] - databit.MAs[0].values[-2]
        if self._allow_trading(databit, trend): #checking if should inore and not enter/exit the market
            #buy or sell
            return current_ma_speed >= self.k*databit.MAs[0].values[-1]

class MAGrowingSpeed1(MAGrowingSpeed0):
    ''':_allow_trading: if close / ema <= 1 + allow_trading_coefficient
    :decide: if speed of ma >= k*ma - buy, if speed of ma < k*ma - sell, else - pass'''
    def __init__(self, k: float=1*10**(-7), allow_trading_coefficient: float=18*10**(-3)):
        self.k = k
        self.allow_trading_coefficient = allow_trading_coefficient
        
    def _allow_trading(self, databit: Candles, trend: list | None = None) -> bool:
        return databit.Close.values[-1] / databit.EMAs[0].values[-1] <= 1 + self.allow_trading_coefficient
    
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        if self._allow_trading(databit, trend):
            return super().decide(databit, trend, predictions, risks)
        else: return False
        
class MAGrowingSpeed2(MAGrowingSpeed0):
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        if self._allow_trading(databit, trend):
            if not any(risks):
                return super().decide(databit, trend, predictions, risks)
        return False
        
class TwoChosenMAsCrossing(DecidingModule):
    def __init__(self, fst: int=0, scd: int=-1):
        assert fst >= 0 
        assert scd >= 0 or scd == -1
        assert scd > fst or scd == -1
        self.least_ma_ind = fst
        self.greater_ma_ind = scd
        
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        assert len(databit.MAs) >= self.greater_ma_ind or self.greater_ma_ind == -1
        return databit.MAs[self.least_ma_ind] > databit.MAs[self.greater_ma_ind]
    
class BollingerBounce(DecidingModule):
    def __init__(self, bounce_up: float=0, bounce_down: float=0):
        self.bounce_up = bounce_up
        self.bounce_down = bounce_down
    
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        if databit.Close <= databit.BollingerBands[0] * (1+self.bounce_up) or databit.Low <= databit.BollingerBands[0] * (1+self.bounce_up): 
            return True
        elif databit.Close >= databit.BollingerBands[2] * (1+self.bounce_down) or databit.High >= databit.BollingerBands[2] * (1+self.bounce_down):
            return False
        else:
            return None
    
class MishmashDecision(DecidingModule):
    def __init__(self, bounce_up: float=0, bounce_down: float=0, greater_ma_ind: int=-1, least_ma_ind: int=0):
        self.bounce_up = bounce_up
        self.bounce_down = bounce_down
        self.greater_ma_ind = greater_ma_ind
        self.least_ma_ind = least_ma_ind
    
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        if not any(risks):
            assert len(databit.MAs) >= self.greater_ma_ind or self.greater_ma_ind == -1
            if any([databit.Close <= databit.BollingerBands[0] * (1+self.bounce_up), 
                    databit.Low <= databit.BollingerBands[0] * (1+self.bounce_up),     # bollinger_bounce thing here
                    databit.MAs[self.least_ma_ind] > databit.MAs[self.greater_ma_ind], # crossing mas thig
                    databit.MAs[self.least_ma_ind] > databit.Close,  # least span ma higher than price
                    trend[0].pred.biased > databit.Close]): #trend taken in account
                return True
            elif any([databit.Close >= databit.BollingerBands[2] * (1+self.bounce_down),
                      databit.High >= databit.BollingerBands[2] * (1+self.bounce_down),# bollinger_bounce part here
                      not databit.MAs[self.least_ma_ind] > databit.MAs[self.greater_ma_ind], # cross mas thig
                      not databit.MAs[self.least_ma_ind] > databit.Close, # least span ma higher than price
                      trend[0].pred.biased < databit.Close]): #trend taken in account
                return False
            else:
                return None
        else:
            return False
        
class MishmashNoAI(DecidingModule):
    def __init__(self, bounce_up: float=0, bounce_down: float=0, greater_ma_ind: int=-1, least_ma_ind: int=0):
        self.bounce_up = bounce_up
        self.bounce_down = bounce_down
        self.greater_ma_ind = greater_ma_ind
        self.least_ma_ind = least_ma_ind
    
    def decide(self, databit: Candles, trend: list[PredictorResponse] | None = None, predictions: list[PredictorResponse] | None = None, risks: list | None = None) -> bool | None:
        if not any(risks):
            assert len(databit.MAs) >= self.greater_ma_ind or self.greater_ma_ind == -1
            if any([databit.Close <= databit.BollingerBands[0] * (1+self.bounce_up), 
                    databit.Low <= databit.BollingerBands[0] * (1+self.bounce_up),     # bollinger_bounce thing here
                    databit.MAs[self.least_ma_ind] < databit.MAs[self.greater_ma_ind], # crossing mas thig
                    databit.MAs[self.least_ma_ind] < databit.Close]):  # least span ma higher than price
                return True
            elif any([databit.Close >= databit.BollingerBands[2] * (1+self.bounce_down),
                      databit.High >= databit.BollingerBands[2] * (1+self.bounce_down),# bollinger_bounce part here
                      not databit.MAs[self.least_ma_ind] < databit.MAs[self.greater_ma_ind], # cross mas thig
                      not databit.MAs[self.least_ma_ind] < databit.Close]): # least span ma higher than price
                return False
            else:
                return None
        else:
            return False