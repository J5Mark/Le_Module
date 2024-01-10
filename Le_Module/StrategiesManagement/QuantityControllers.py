import random as rd
from Utilities.FinUtils import Candles
import numpy as np
from numpy import ndarray
from abc import ABC, abstractmethod

class QuantityController(ABC):
    '''this module decides on the amount of security lots to buy or sell.'''
    @abstractmethod
    def decide_quantity(self, databit: Candles, direction: bool, trend: list | None = None, predictions: np.ndarray | None = None, risks: list | None = None) -> int:
        pass

class AIQuantityControllerTPR1(QuantityController):
    def decide_quantity(self, direction: bool, databit: Candles, predictions: ndarray | None = None, risks: list | None = None) -> int:
        if any(risks): return -1
        if direction:
            N = 1
            if predictions[0][0][1] > databit[-1].Close*1.0005:
                for p in range(1, len(predictions)):
                    if predictions[p][0][1] > predictions[p-1][0][1]:
                        N = p+1
        else: N = -1 #-1 meaning all
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