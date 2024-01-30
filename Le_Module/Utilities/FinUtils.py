from __future__ import annotations
import numpy as np
import pandas as pd
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now
from datetime import timedelta
from dataclasses import dataclass
from PredictorsManagement.pm import *
from Utilities.DataClasses import *

#Ive decided to keep technical indicators functions more-less the same in what arguments they take

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

def get_training_data(indicators: Candles, len_of_sample : int = 5, len_of_label : int = 1, scope : int | None=None, predictable: str | market_condition='close') -> tuple:
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
        if isinstance(predictable, str):
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
        elif isinstance(predictable, market_condition):  ###################
            for i in range(len(asdf)-len_of_sample-1):
                ins = asdf[:][i:i+len_of_sample]
                pic = np.reshape(ins.values, (len_of_sample, asdf.shape[1]))
                y = predictable(indicators[i+1])
                training.append(pic)
                labels.append(y)
        training = np.array(training)
        labels = np.array(labels)
        return (training, labels)
    except KeyError:
        print(f'{predictable} is not a name of an indicator, or it is just spelled incorrectly\n should be in: \n{[i for i in cols]}')
    

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