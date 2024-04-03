from __future__ import annotations
import numpy as np
import pandas as pd
import dataclasses
from dataclasses import dataclass
import math
from tinkoff.invest import MoneyValue, Quotation
import keras
from abc import ABC, abstractmethod

class WrongDimsNdArrayError(Exception):
    def __init__(self, correct_dims: tuple, provided_dims: tuple):
        self.msg = f'np.ndarray is of unfit shape.\nShape of provided np.ndarray: {provided_dims}\nCorrect shape: {correct_dims}'
        super().__init__(self.msg)
        
class AppendingImpossibleError(Exception):
    def __init__(self, ind: Indicator | Candles, val):
        if isinstance(ind, Indicator):
            self.msg = f'{val} (type: {val.__class__}) cannot be appended to {ind.name} indicator.\nValues provided:\n   {val}\n   {ind}'
        elif isinstance(ind, Candles):
            self.msg = f'{val} (type: {val.__class__}) cannot be appended to {ind.__class__} object.\nValues provided:\n   {val}\n   {ind}'
        super().__init__(self.msg)
        
class DifferentIndicatorsNamesError(Exception):
    def __init__(self, ind1: Indicator, ind2: Indicator):
        self.msg = f'Appending {ind2.name} to {ind1.name} makes no sense and is useless'
        super().__init__(self.msg)

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
    
    def append(self, vals: Indicator | list | np.ndarray | float):
        match vals:
            case Indicator():
                if self.name == vals.name:
                    self.values = np.array(self.values.tolist() + vals.values.tolist())
                else:
                    raise DifferentIndicatorsNamesError(self, vals)
            case list():
                self.values = np.array(self.values.tolist() + vals)
            case np.ndarray():
                self.values = np.array(self.values.tolist() + vals.tolist())
            case float():
                self.values = np.array(self.values.tolist() + [vals])
            case int():
                self.values = np.array(self.values.tolist() + [float(vals)])
            case None:
                return
            case _:
                raise AppendingImpossibleError(self, vals)
                
            
@dataclass
class Candles:
    '''An array of candles'''
    Open: Indicator
    Close: Indicator
    High: Indicator
    Low: Indicator 
    Volume: Indicator | None = None

    MAs: list[Indicator] | None = None #first goes the lower span MA
    EMAs: list[Indicator] | None = None #same thing here
    MACDhist: Indicator | None = None
    ATR: Indicator | None = None
    BollingerBands: list[Indicator] | None = None
    Volatility: Indicator | None = None

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
    
    def as_dataframe(self) -> pd.DataFrame: #create dataframe with a column for each indicator. Make sure the lengthof the frame is the minimum of all lengths #drop duplicate columns
        names_of_attributes = self.__dict__.keys()
        lens = []
        for name in names_of_attributes: 
            attribute = self.__getattribute__(name)
            if isinstance(attribute, Indicator):
                lens.append(len(attribute.values[attribute.values != np.array(None)]))
            elif isinstance(attribute, list):
                for each in attribute:
                    lens.append(len(each.values[each.values != np.array(None)]))
                    
        m = min(lens) #the minimal length defined here so that i dont have to mention it through min(lens) all the time
        df = pd.DataFrame([]) #lengths of all indicators are figured out and a dataframe is created
        
        list_of_indicators: list[Indicator] = []
        for name in names_of_attributes:
            the_attribute = self.__getattribute__(name)
            if isinstance(the_attribute, list):
                for each in the_attribute:
                    list_of_indicators.append(each[-m:])
            elif isinstance(the_attribute, Indicator):
                list_of_indicators.append(the_attribute[-m:])  #here all of the attributes are sorted based on their type
                
        for each in list_of_indicators:
            df[f'{each.name} span: {each.span}' if each.span != [0] else f'{each.name}'] = each.values
            
        df = df.loc[:,~df.columns.duplicated()].copy()
        
        return df
    
    def append(self, c: Candles | Indicator):  ###later plss
        match c:
            case Candles():
                pass
            case Indicator():
                pass
            case _:
                raise AppendingImpossibleError(self, c)
    
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
    
@dataclass
class ParameterControllerData:
    model: keras.Sequential | None=None
    training_data: tuple | None=None
    def __post_init__(self):
        if self.model == None:
            pass
        
@dataclass
class Collision:
    """
        :level: as if a ball bounces off of a wall, this is the less volatile line between the two colliding
        :line: same analogy, the more volatile line
        if side == True the line bounced up from the level else: bounced down
        if side is None no collision occured during a given span of time
        :ago: amount of candles passed since the end of collision(current candle included)
    """    
    side: bool | None
    line: str
    level: str
    ago: int | None
    
    def __post_init__(self):
        if None in (self.side, self.ago): self.side = None; self.ago = None
        
class DataPreparationInstructions(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, db: Candles) -> Candles:
        pass
    
@dataclass    
class Order:
    direction: bool
    amount: int
    price: float
    order_id: str
    order_type: int
    figi: str
    
@dataclass
class Asset:
    figi: str
    name : str | None=None
    lot_size: int | None=None
