from __future__ import annotations
import numpy as np
import pandas as pd
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now
from datetime import timedelta
import dataclasses
from dataclasses import dataclass
import math
from tinkoff.invest import MoneyValue, Quotation

#Ive decided to keep technical indicators functions more-less the same in what arguments they take
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
        return Indicator(name=self.name, values=self.values[key], span=self.span)

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


def calcMACD(data : Candles) -> Indicator:  
    '''MACD histogram is one of the main indicators for assessing the "impulse" of the market movement'''
    prices = pd.Series(data.Close.values)
    indicator = prices.ewm(span=12, adjust=False, min_periods=12).mean() - prices.ewm(span=26, adjust=False, min_periods=26).mean()
    signal = indicator.ewm(span=9, adjust=False, min_periods=9).mean()
    d = indicator - signal
    return Indicator(name='MACD histogram', values=np.reshape(d.values[33:], (len(d)-33)), span=[26, 12, 9])

def calcMA(data : Candles, span : int) -> Indicator:
    '''Moving average of a value over a certain span'''
    mean = []
    cvals = data.Close.values
    for e in range(len(cvals[span:])):
        mean.append(np.mean(cvals[e-span:e]))
    return Indicator(name='moving average', values=np.array(mean)[span:], span=[span])

def calcATR(data : Candles) -> Indicator:
    '''ATR is one of technical indicators mainly used for managing trading risks and assessing current market volatility'''
    atr = [0]
    cvals = data.Close.values
    hvals = data.High.values
    lvals = data.Low.values
    for i in range(1, len(cvals)):
        pre_tr = [hvals[i] - lvals[i], abs(hvals[i] - cvals[i-1]), abs(lvals[i] - cvals[i-1])]
        atr.append((atr[-1]*13 + max(pre_tr))/14)
    return Indicator(name='average true range', values=np.array(atr), span=[0])

def calcBollinger_bands(data : Candles, span : int=20) -> list[Indicator]: 
    '''Bollinger bands are useful for understanding the gap in which potential security price movement may occur'''
    middle_band = calcMA(data, span=span)
    middle_band.name = 'middle bollinger band'
    middle_band.values = middle_band.values
    cvals = data.Close.values
    stds = []
    for i in range(span, len(cvals)):
        stds.append(np.std(cvals[i-span:i]))
    stds = np.array(stds[span:])
    upper_band = Indicator(name='upper bollinger band', values=middle_band.values + 2*stds, span=[span])
    lower_band = Indicator(name='lower bollinger band', values=middle_band.values - 2*stds, span=[span])
    return [lower_band, middle_band, upper_band]

def createdataset_yfinance(df : Candles) -> Candles:  ## currently working on this
    '''use this function for preprocessing if working with a yf dataset'''
    indicators = Candles(Open = Indicator(name='open', values=df.Open.values, span=[0]), 
                        Close = Indicator(name='close', values=df.Close.values, span=[0]), 
                        High = Indicator(name='high', values=df.High.values, span=[0]),
                        Low = Indicator(name='low', values=df.Low.values, span=[0]),
                        Volume= Indicator(name='volume', values=df.Volume.values, span=[0]),
                        MACDhist = calcMACD(df),
                        MAs = list[calcMA(df, 20), calcMA(df, 50)],
                        ATR = calcATR(df),
                        BollingerBands = calcBollinger_bands(df))
    return indicators

def createdataset_tinkoff(df : Candles) -> Candles:
    '''use this function for preprocessing data if working with Tinkoff API'''
    df.MACDhist = calcMACD(df)
    df.MAs = [calcMA(df, 20), calcMA(df, 50)]
    df.ATR = calcATR(df)
    df.BollingerBands = calcBollinger_bands(df)
    return df

def get_training_data(indicators: Candles, len_of_sample : int = 5, len_of_label : int = 1, scope : int = 1, predictable: str='close') -> tuple:
    '''Split your Candles dataset into samples and labels.

    :scope: is how far your model is going to predict. 
    Ex: with scope=1 it will predict right the next value of the predictable indicator
    Ex: with scope=2 it will predict the predictable indicator of the candle after the upcoming one
    
    :predictable: is the name of the technical indicator the model will be predicting. Should be within the list'''
    try:
        training = []
        labels = []
        
        asdf = indicators.as_dataframe()
        cols = asdf.columns
        
        for i in range(len(asdf)-len_of_sample-len_of_label+1-scope):
            ins = pd.DataFrame([])
            ins = asdf[:][i:i+len_of_sample]  
        
            if len_of_label == 1:          
                y = asdf.iloc[i+len_of_sample+scope-1, cols.get_loc(predictable)]
            else:
                y = asdf.iloc[i+len_of_sample+scope-1:i+len_of_sample+scope-1+len_of_label, cols.get_loc(predictable)]
            pic = np.reshape(ins.values, (len_of_sample, asdf.shape[1]))

            training.append(pic)
            labels.append(y)

        training = np.array(training)
        labels = np.array(labels)
        return (training, labels)
    except KeyError:
        print(f'{predictable} is not a name of an indicator, or it is just written incorrectly\n should be in: \n{[i for i in cols]}')
    except:
        print('something else happened')
    

def get_data_tinkoff(TOKEN : str, FIGI : str, period : int=12, interval : CandleInterval=CandleInterval.CANDLE_INTERVAL_5_MIN) -> Candles:
    '''Get the candles of the ticker of interest with Tinkoff API'''
    with Client(TOKEN) as client:
        open, close, high, low, volume = [], [], [], [], []

        for candle in client.get_all_candles(
            figi=FIGI,
            from_=now() - timedelta(days=period),
            to=now(),
            interval=interval
        ):
            open.append(Money(candle.open).units + Money(candle.open).nano/(10**9)) 
            close.append(Money(candle.close).units + Money(candle.close).nano/(10**9))
            high.append(Money(candle.high).units + Money(candle.high).nano/(10**9))
            low.append(Money(candle.low).units + Money(candle.low).nano/(10**9))
            #open.append(candle.open.units + candle.open.nano * 10**(-9))
            #close.append(candle.close.units + candle.close.nano * 10**(-9))
            #high.append(candle.high.units + candle.high.nano * 10**(-9))
            #low.append(candle.low.units + candle.low.nano * 10**(-9))
            volume.append(candle.volume)        
    
    open = Indicator(name='open', values=np.array(open), span=[0])
    close = Indicator(name='close', values=np.array(close), span=[0])
    high = Indicator(name='high', values=np.array(high), span=[0])
    low = Indicator(name='low', values=np.array(low), span=[0])
    volume = Indicator(name='volume', values=np.array(volume), span=[0])

    dt = Candles(Open=open, Close=close, High=high, Low=low, Volume=volume)

    return dt

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