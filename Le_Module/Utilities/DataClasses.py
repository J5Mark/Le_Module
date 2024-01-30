from __future__ import annotations
import numpy as np
import pandas as pd
import dataclasses
from dataclasses import dataclass
import math
from tinkoff.invest import MoneyValue, Quotation
from PredictorsManagement.pm import *

@dataclass(init=False, order=True)
class Money:
    units: int
    nano: int
    MOD: int = 10 ** 9

    def __init__(self, value: int | float | Quotation | MoneyValue, nano: int = None):
        if nano:
            assert isinstance(value, int), 'if nano is present, value must be int'
            assert isinstance(nano, int), 'nano must be int'
            self.units = value
            self.nano = nano
        else: 
            match value:
                case int() as value:
                    self.units = value
                    self.nano = 0
                case float() as value:
                    self.units = int(math.floor(value))
                    self.nano = int((value - math.floor(value)) * self.MOD)
                case Quotation() | MoneyValue() as value:
                    self.units = value.units
                    self.nano = value.nano
                case _:
                    raise ValueError(f'{type(value)} is not supported as initial value for Money')

    def __float__(self):
        return self.units + self.nano / self.MOD

    def to_float(self):
        return float(self)

    def to_quotation(self):
        return Quotation(self.units, self.nano)

    def to_money_value(self, currency: str):
        return MoneyValue(currency, self.units, self.nano)

    def __add__(self, other: Money) -> Money:
        print(self.units + other.units + (self.nano + other.nano) // self.MOD)
        print((self.nano + other.nano) % self.MOD)
        return Money(
            self.units + other.units + (self.nano + other.nano) // self.MOD,
            (self.nano + other.nano) % self.MOD
        )

    def __neg__(self) -> Money:
        return Money(-self.units, -self.nano)

    def __sub__(self, other: Money) -> Money:
        return self + -other

    def __mul__(self, other: int) -> Money:
        return Money(self.units * other + (self.nano * other) // self.MOD, (self.nano * other) % self.MOD)

    def __str__(self) -> str:
        return f'<Money units={self.units} nano={self.nano}>'

@dataclass
class Indicator:
    name: str
    values: np.ndarray | float
    span: list[int] | None = None

    def __getitem__(self, key):
        return Indicator(name=self.name, values=self.values[key] if isinstance(self.values[key], list) or isinstance(self.values[key], np.ndarray) else np.array([self.values[key]]), span=self.span)
    
    def __eq__(self, __value: Indicator) -> bool:
        if isinstance(__value, Indicator):
            return self.values[-1] == __value.values[-1]
        elif isinstance(__value, list) or isinstance(__value, tuple):
            return self.values[-1] == __value[-1]
        else:
            return self.values[-1] == __value
        
    def __gt__(self, __value: Indicator) -> bool:
        if isinstance(__value, Indicator):
            return self.values[-1] > __value.values[-1]
        elif isinstance(__value, list) or isinstance(__value, tuple):
            return self.values[-1] > __value[-1]
        else:
            return self.values[-1] > __value
        
    def __lt__(self, __value: Indicator) -> bool:
        if isinstance(__value, Indicator):
            return self.values[-1] < __value.values[-1]
        elif isinstance(__value, list) or isinstance(__value, tuple):
            return self.values[-1] < __value[-1]
        else:
            return self.values[-1] < __value
    
    def __ge__(self, __value: Indicator) -> bool:
        if isinstance(__value, Indicator):
            return self.values[-1] >= __value.values[-1]
        elif isinstance(__value, list) or isinstance(__value, tuple):
            return self.values[-1] >= __value[-1]
        else:
            return self.values[-1] >= __value
    
    def __le__(self, __value: Indicator) -> bool:
        if isinstance(__value, Indicator):
            return self.values[-1] <= __value.values[-1]
        elif isinstance(__value, list) or isinstance(__value, tuple):
            return self.values[-1] <= __value[-1]
        else:
            return self.values[-1] <= __value
    
    def __len__(self) -> int:
        return len(self.values)
    
    def __mul__(self, __value: Indicator | float) -> Indicator:
        if isinstance(__value, Indicator):
            leastlen = min(len(self), len(__value))
            return Indicator(name=f'{self.name}*{__value.name}', values=np.array(self.values[-leastlen:])*np.array(__value.values[-leastlen:]), span=self.span+__value.span)
        else:
            return Indicator(name=f'{self.name}*{__value}', values=np.array(self.values)*__value, span=self.span)
        
    def __add__(self, __value: IndentationError | float):
        if isinstance(__value, Indicator):
            leastlen = min(len(self), len(__value))
            return Indicator(name=f'{self.name}+{__value.name}', values=np.array(self.values[-leastlen:])+np.array(__value.values[-leastlen:]), span=self.span+__value.span)
        elif isinstance(__value, float):
            return Indicator(name=f'{self.name}+{__value}', values=np.array(self.values)+__value, span=self.span)
        
    def __sub__(self, __value: IndentationError | float):
        if isinstance(__value, Indicator):
            leastlen = min(len(self), len(__value))
            return Indicator(name=f'{self.name}-{__value.name}', values=np.array(self.values[-leastlen:])-np.array(__value.values[-leastlen:]), span=self.span+__value.span)
        elif isinstance(__value, float):
            return Indicator(name=f'{self.name}-{__value}', values=np.array(self.values)-__value, span=self.span)
        
    def __truediv__(self, __value: Indicator | float):
        if isinstance(__value, Indicator):
            leastlen = min(len(self), len(__value))
            return Indicator(name=f'{self.name}/{__value.name}', values=np.array(self.values[-leastlen:])/np.array(__value.values[-leastlen:]), span=self.span+__value.span)
        elif isinstance(__value, float):
            return Indicator(name=f'{self.name}/{__value}', values=np.array(self.values)/__value, span=self.span)
        
    def __iadd__(self, __value: Indicator | float):
        self = self + __value
        return self
    
    def __isub__(self, __value: Indicator | float):
        self = self - __value
        return self
            
@dataclass
class Candles:
    '''An array of candles'''
    Open: Indicator
    Close: Indicator
    High: Indicator
    Low: Indicator 
    Volume: Indicator | None = None

    MAs: list[Indicator] | None = None #first goes the lower span MA
    MACDhist: Indicator | None = None
    ATR: Indicator | None = None
    BollingerBands: list[Indicator] | None = None

    FIGI: str | None = None
    SecurityName: str | None = None

    def __iter__(self):
        yield from dataclasses.asdict(self).values()

    def __getitem__(self, key):
        _ = Candles(0, 0, 0, 0, 0)
        for name in _.__dict__.keys():
            if isinstance(self.__getattribute__(name), Indicator):     
                _.__setattr__(name, self.__getattribute__(name)[key])
            elif isinstance(self.__getattribute__(name), list):
                _.__setattr__(name, [e[key] for e in self.__getattribute__(name)])
                    
            else:
                _.__setattr__(name, self.__getattribute__(name))
        
        return _
    
    def as_nparray(self) -> np.ndarray:
        return self.as_dataframe().to_numpy()

    def as_dict(self) -> dict:
        return self.__dict__
    
    def as_dataframe(self) -> pd.DataFrame:
        all_values = []
        for name in self.__dict__.keys(): 
            if isinstance(self.__getattribute__(name), Indicator):
                all_values += [self.__getattribute__(name)]
            elif isinstance(self.__getattribute__(name), list):
                for each in self.__getattribute__(name):
                    all_values += [each]
            
        lens = [len(g.values) for g in all_values if g != None]
        df = pd.DataFrame([])

        for each in [self.__getattribute__(name) for name in self.__dict__.keys() if not isinstance(self.__getattribute__(name), list) and not isinstance(self.__getattribute__(name), str)]:
            if each != None:
                df[each.name] = each.values[-min(lens):]
        if self.BollingerBands != None and self.MAs != None:
            if [i for i in self.BollingerBands][1].span[0] not in [g.span[0] for g in self.MAs]:
                for each in [self.MAs, self.BollingerBands]:
                    for t in each:
                        ma = t.name == 'moving average'
                        df[t.name + ' ' + int(ma)*str(t.span[0])] = t.values[-min(lens):]
            else:
                for each in [[i for i in self.MAs if self.MAs != None], [[i for i in self.BollingerBands if self.BollingerBands != None][0], [i for i in self.BollingerBands if self.BollingerBands != None][2]]]:
                    for t in each:
                        ma = t.name == 'moving average'
                        df[t.name + ' ' + int(ma)*str(t.span[0])] = t.values[-min(lens):]
            
        return df
    
@dataclass
class Decision:
    '''dataclass for trading decisions.
      :direction: True - buy, False - sell
      :amount: how many lots to buy/sell, -1 for all in access
      :type: type of the order, 0 - market, 1 - stoploss, 2 - takeprofit'''
    direction: bool
    amount: int
    # These ones are really important for real trading, not for backtesting:
    type: int=0
    price: float=-1
    
@dataclass
class DecisionsBatch:
    market: Decision | None=None
    stop_loss: Decision | None=None
    take_profit: Decision | None=None    
    
@dataclass
class DatabitsBatch:
    for_predictors: Candles | None=None
    for_trendviewers: Candles | None=None
    for_risk_managers: Candles | None=None