from __future__ import annotations
from ccxt.base import exchange
from grpc import xds_channel_credentials
import numpy as np
import pandas as pd
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now
from datetime import timedelta
from PredictorsManagement.pm import *
from Utilities.DataClasses import *
import ccxt

#Ive decided to keep technical indicators functions more-less the same in what arguments they take

def calcCtoCVolatility(data : Candles, span : int):
    '''Close-to-Close volatility'''
    assert span > 1
    cvals = pd.Series(data.Close.values)
    ma = cvals.rolling(span).mean().dropna().to_numpy()
    volatilities = []
    for i in range(span, len(data.Close.values)):
        volatilities.append(np.sqrt(sum(data.Close.values[i-span:i]-ma[i-span])**2/(span-1)))

    return Indicator(name='Close-to-Close volatility', values=np.array(volatilities), span=[span]) #strange thing ngl

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

def calcEntropy(c: Candles, span: int, n_of_levels: int = 40, indicator: str='close') -> float:
    in_question = None
    for ind in list(c.__dict__.values()):
        if indicator == ind.name:
            in_question: Indicator = ind
            break
    if in_question is None: raise WrongIndicatorSelectedError(indicator, c)

    borders = (min(in_question.values[-span:]), max(in_question.values[-span:]))
    s = (borders[1]-borders[0])/n_of_levels
    levels = np.arange(start=borders[0], stop=borders[1]+s, step=s)
    kinda_probabilities = [sum(np.bitwise_and(in_question.values >= levels[i-1], in_question.values < levels[i]))/len(in_question.values) for i in range(len(levels))]
    entropy = -sum([p*np.log2(p) for p in kinda_probabilities[1:] if p != 0])

    return entropy

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


def get_candles_tinkoff(TOKEN : str, FIGI : str, period : int=12, interval : CandleInterval=CandleInterval.CANDLE_INTERVAL_DAY) -> Candles:
    '''Get the candles of the ticker of interest with Tinkoff API'''
    with Client(TOKEN) as client:
        open, close, high, low, volume = [], [], [], [], []
        time = []
        for candle in client.get_all_candles(
            figi=FIGI,
            from_=now() - timedelta(days=period),
            to=now(),
            interval=interval
        ):
            time.append(candle.time)
            open.append(Money(candle.open).units + Money(candle.open).nano/(10**9))
            close.append(Money(candle.close).units + Money(candle.close).nano/(10**9))
            high.append(Money(candle.high).units + Money(candle.high).nano/(10**9))
            low.append(Money(candle.low).units + Money(candle.low).nano/(10**9))
            volume.append(candle.volume)

    open = Indicator(name='open', values=np.array(open), span=[0])
    close = Indicator(name='close', values=np.array(close), span=[0])
    high = Indicator(name='high', values=np.array(high), span=[0])
    low = Indicator(name='low', values=np.array(low), span=[0])
    volume = Indicator(name='volume', values=np.array(volume), span=[0])

    c = Candles(Open=open, Close=close, High=high, Low=low, Volume=volume, Time=time, Symbol=FIGI)

    return c

def get_candles_ccxt(pair: str, period: int, interval: str, exchange: str = 'binance', timeout: int=10_000) -> Candles:
    #some arguments may make less sense because i whanted get_candles_ccxt and get_candles_tinkoff to have the same arguments. The tinkoff function just appeared earlier
    '''
        :pair: name of currency pair, for example BTC/USD
        :period: how many candles to fetch
        :timeout: timeout in milisecs
        :interval: aka timeframe, should be in format like 1h
        :exchange: name of exchange yoou want to fetch data from. To see which ones are available use ccxt.exchanges
        Also ccxt tells time in milisecs since 1st january of 1970, so keep in mind.
    '''
    assert exchange in ccxt.exchanges, f'ccxt thinks there is no such exchange as {exchange}'

    xch_id = exchange
    xch_class = getattr(ccxt, xch_id)
    xch = xch_class({
        'timeout' : timeout,
        'interval': interval
    })

    assert xch.has['fetchOHLCV'], f'echange {exchange} doesn appear to have a fetchOHLCV method, which is essential'

    open, close, high, low, volume, time = [], [], [], [], [], []
    for candle in xch.fetchOHLCV(pair, limit=period):
        time.append(candle[0])
        open.append(candle[1])
        high.append(candle[2])
        close.append(candle[3])
        volume.append(candle[4])
    open = Indicator(name='open', values=np.array(open), span=[0])
    close = Indicator(name='close', values=np.array(close), span=[0])
    high = Indicator(name='high', values=np.array(high), span=[0])
    low = Indicator(name='low', values=np.array(low), span=[0])
    volume = Indicator(name='volume', values=np.array(volume), span=[0])

    c = Candles(Open=open, Close=close, High=high, Low=low, Volume=volume, Time=time, Symbol=pair)

    return c

def find_levels(candles: Candles, windowlen: int=100, N: int=3, area: float=0.005) -> list[PriceLevel]:
    '''
        This feature is experimental and may not work as expected
    '''
    levels = []
    mean_price = np.mean(candles.Close.values[-windowlen:])
    for i in range(len(candles.Close.values)-windowlen-N, len(candles.Close.values), N//2):
        current_high_window = candles.High.values[i:i+N]
        current_low_window = candles.Low.values[i:i+N]
        local_max, lmi = max(current_high_window), np.where(current_high_window == max(current_high_window))[0][0] #a value and an index where it is in the current window
        local_low, lli = min(current_low_window), np.where(current_low_window == min(current_low_window))[0][0]
        lm_delta = min([local_max - candles.Close.values[i + lmi], local_max - candles.Open.values[i + lmi]])
        ll_delta = min([candles.Close.values[i + lli] - local_low, candles.Open.values[i + lli] - local_low])
        lm = local_max - 0.25*lm_delta
        ll = local_low + 0.25*ll_delta
        levels.append((lm - (area)*mean_price, lm + (area)*mean_price))
        levels.append((ll - (area)*mean_price, ll + (area)*mean_price))
    intersections = []
    for i in range(len(levels)):
        current_set_of_close_diapazons = [levels[i]]
        for e in range(i, len(levels)):
            if (levels[i][1] > levels[e][0] and levels[i][1] < levels[e][1]) or (levels[i][0] > levels[e][0] and levels[i][0] < levels[e][1]):
                betw_closest = min([abs(levels[i][0] - levels[e][1]), abs(levels[i][1] - levels[e][0])])
                betw_farest = max([abs(levels[i][0] - levels[e][1]), abs(levels[i][1] - levels[e][0])])
                if betw_farest == betw_closest or betw_closest / betw_farest >= 0.5:
                    current_set_of_close_diapazons.append(levels[e])
                intersections.append(current_set_of_close_diapazons)

    unique_intersections_ends = set([i[-1] for i in intersections])
    unique_intersections = [max([inter for inter in intersections if inter[-1] == current_end], key=len) for current_end in unique_intersections_ends] #changed to max here
    real_areas = [(round(min([y[0] for y in u]), 6), round(min([y[1] for y in u]), 6)) for u in unique_intersections]
    pricelevels = []
    for area in real_areas:
        high_indices = [i for i in range(windowlen) if area[0] <= candles.High.values[-windowlen + i] <= area[1]]
        low_indices = [i for i in range(windowlen) if area[0] <= candles.Low.values[-windowlen + i] <= area[1]]
        pricelevels.append(PriceLevel(price_area=area, occurance=len(high_indices) + len(low_indices),
                                last_encounter=min([windowlen - ([windowlen] + low_indices)[-1], windowlen - ([windowlen] + high_indices)[-1]])))

    return pricelevels


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

def find_local_extrema(line: Indicator | np.ndarray | list, extr_type: str='all', N: int=2) -> dict[list[tuple]]:
    '''
        Outputs a dict of lists of (x, y) coordinates of extrema of a line

        :extr_type: specifies what should the algorithm search for:
        "min" for minima
        "max" for maxima
        "all" for both

        :N: is the amount of neighbors an extremum has to be greater/less than
        Also probs jus dont use this function. Its under maintenance
    '''
    assert N >= 2

    match line:
        case Indicator():
            d = line.values
        case np.ndarray():
            d = line
        case list():
            d = np.array(line)
        case _:
            raise TypeError("Data of unexpected type inserted. Should be in : (Indicator, np.ndarray, list)")

    if len(d.shape) != 1: raise WrongDimsNdArrayError(len(d), d.shape)
    minima = []
    maxima = []
    match extr_type:
        case 'min':
            minima = np.where((d[N-1:-1] < d[0:-N]) * (d[N-1:-1] < d[N:]))[0] + 1
        case 'max':
            maxima = np.where((d[N-1:-1] > d[0:-N]) * (d[N-1:-1] > d[N:]))[0] + 1
        case 'all':
            maxima = np.where((d[N-1:-1] > d[0:-N]) * (d[N-1:-1] > d[N:]))[0] + 1
            minima = np.where((d[N-1:-1] < d[0:-N]) * (d[N-1:-1] < d[N:]))[0] + 1
        case _:
            raise AssertionError(f'non-existent type inserted({type}), should be in: (min, max, all)')

    return {'maxima' : [d[coord] if coord in maxima else None for coord in range(len(d))],
            'minima' : [d[coord] if coord in minima else None for coord in range(len(d))]} ## have to do it by myself :((
