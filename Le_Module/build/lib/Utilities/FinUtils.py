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
    cvals = pd.Series(data.Close.values)
    ma = cvals.rolling(span).mean()
    return Indicator(name='moving average', values=ma.dropna().to_numpy(), span=[span])

def calcEMA(data : Candles, span : int) -> Indicator:
    prices = pd.Series(data.Close.values)
    ema = prices.ewm(span=span, adjust=False, min_periods=span).mean()
    return Indicator(name='exponential moving average', values=np.array(ema), span=[span])

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

def calcVolatility(c: Candles, span: int=20) -> Indicator:
    vols = []
    for i in range(span, len(c.Close)):
        std = np.std(c.Close.values[i-span:i])
        vols.append(std*(span**0.5))
    
    return Indicator(name='volatility', values=np.array(vols), span=[span])

def calcBollinger_bands(data : Candles, span : int=20) -> list[Indicator]: 
    '''Bollinger bands are useful for understanding the gap in which potential security price movement may occur'''
    middle_band = calcMA(data, span=span)
    middle_band.name = 'middle bollinger band'
    middle_band.values = middle_band.values
    cvals = data.Close.values
    stds = []
    for i in range(len(cvals)):
        stds.append(np.std(cvals[i-span:i]))
    stds = np.array(stds)
    minlen = min(len(stds), len(middle_band))
    upper_band = Indicator(name='upper bollinger band', values=middle_band.values[-minlen:] + 2*stds[-minlen:], span=[span])
    lower_band = Indicator(name='lower bollinger band', values=middle_band.values[-minlen:] - 2*stds[-minlen:], span=[span])
    return [lower_band, middle_band, upper_band]

def createdataset_yfinance(df : Candles) -> Candles: 
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
    df.MAs = [calcMA(df, 10), calcMA(df, 20)]
    df.EMAs = [calcEMA(df, 10), calcEMA(df, 20)]
    df.ATR = calcATR(df)
    df.BollingerBands = calcBollinger_bands(df)
    return df

def get_training_data(indicators: Candles, len_of_sample : int = 5, len_of_label : int = 1, scope : int | None=None, predictable: str | market_condition='close') -> tuple[np.ndarray]:
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
        elif isinstance(predictable, market_condition): 
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

def find_extremums(candles: Candles, windowlen: int=100, the_N: int=1): ############################### guess ill just finish this a bit later
    extremums = {'peaks': [], 'dips': []}
    for i in range(the_N, len(candles)-the_N, the_N):
        mn = candles[candles.Low.values.index(min(candles[i-the_N : i+the_N].Low.values))]
        mx = candles[candles.High.values.index(max(candles[i-the_N : i+the_N].Low.values))]
        extremums['dips'].append(mn)
        extremums['peaks'].append(mx)
        for e in range(i-the_N, i+the_N):
            pass
            

def find_levels(candles: Candles, windowlen: int=100, the_N: int=1):
    pass

def detect_last_collision(line: Indicator, level: Indicator, strictness: float = 0.01) -> Collision:
    '''
        detects if a given line has bounced off of a given level
        :line: - the line collision of which is being detected
        :level: - the level collision with which is being detected
    '''
    assert len(line) == len(level)

    mask = line.values > level.values * (1 + (-1 if line > level else 1)*-strictness)
    has_bounced = len(set(mask)) == 2
    if has_bounced:
        current_condition = mask[-1]
        
        r = list(reversed(mask))
        end_of_collision = r.index(~current_condition)
        #beginning_of_collision = list(r[end_of_collision:]).index(current_condition) if len(set(r[end_of_collision:])) > 1 else None
        
        side = current_condition and not r[end_of_collision]
        
        return Collision(side, f'{line.name} {line.span}', f'{level.name} {level.span}', ago=end_of_collision+1)
    
    return Collision(side=None, line=f'{line.name} {line.span}', level=f'{level.name} {level.span}', ago=None)

def all_about_drawdowns(pnl):
    ending = np.argmax(np.maximum.accumulate(pnl) - pnl)
    beginning = np.argmax(pnl[:ending])
    MDD = (pnl[beginning] - pnl[ending]) / pnl[beginning]
    if MDD != 0:
        PDD = (pnl[-1]-pnl[0]) / (MDD * pnl[beginning])
    else:
        PDD = None
    return MDD, PDD, beginning, ending

def sharpe_ratio(pnls, riskfree=0):
    std = np.std(np.array(pnls) - pnls[0])
    if std != 0:
        sharpe_ratio = (pnls[-1] - riskfree*pnls[0]/100)/std
    else:
        sharpe_ratio = None
    return sharpe_ratio

