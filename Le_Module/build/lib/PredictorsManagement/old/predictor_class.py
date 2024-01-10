import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import alive_progress
import time
from tinkoff.invest import CandleInterval, Client
from tinkoff.invest.utils import now
from Utilities import money
from datetime import timedelta

#predictor class
#training_data = [np.array(training), np.array(labels)], lyrs = [layers.], raw_data - indicators(dataframe)
class predictor:
    def __init__(self,  lrs=None, model_type='standard', optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.MeanSquaredError(), scope=1): #it's a regression model so no accuracy here
        self.optimizer = optimizer 
        self.loss = loss
        self.scope = scope
        self.lrs = lrs
        if model_type == 'standard':
            self.model = tf.keras.Sequential()
            for each in self.lrs: self.model.add(each)
            self.model.compile(optimizer = self.optimizer, loss = self.loss)
          
        else:
            self.model = model_type #not a str definetely

    def train(self, training_data, labels, epochs=100, verbose=True, callbacks = None):
       self.model.fit(x=training_data, y=labels, epochs=epochs, shuffle=True, verbose=verbose, callbacks=callbacks, validation_split=0.1, batch_size=32, use_multiprocessing=True)

    def examine_bias(self, raw_data, training_data, labels):
      predicts = [] 
      biases = []
      with alive_progress.alive_bar(len(labels)-1, force_tty=True, spinner='radioactive') as bar: 
        for i in range(len(labels)-1):
          time.sleep(0.005)
          prediction = self.pred(training_data[i:i+1])
          predicts.append(prediction)
          biases.append((labels[i] - prediction)/prediction)
          bar()

      predicts = np.array(predicts)
      predicts = np.append(np.array([None]*(len(raw_data) - len(training_data) + 1)), predicts)#np.reshape(predicts, (predicts.shape[0], 1)))
      positive = [i for i in biases if i < 0]
      negative = [i for i in biases if i > 0]
      
      for each in [positive, negative]:
        if len(each) == 0:
          each.append(0)

      avg_positive = sum(positive)/len(positive)
      avg_negative = sum(negative)/len(negative)
      self.bias = (avg_positive, avg_negative) #estimation of how pessimistic/optimistic the model is
    
    def pred(self, data):
      return self.model.predict(data, verbose=0)
    
    def make_prediction(self, data):
      p = self.pred(data) 
      return (p+p*self.bias[0], p, p+p*self.bias[1])
    
    def see_performance(self, ins, training_data): #dataframe and array of data like after createdataset
      ps = [] 
      with alive_progress.alive_bar(len(training_data)-1, force_tty=True, spinner='radioactive') as bar:
        for i in range(len(training_data)-1):
          p = self.make_prediction([training_data[i]])
          ps.append(p)
          bar()
      
      plt.plot(ps)
      plt.plot(ins.close.values)
      plt.show()         


############################################################################################################################

## methods for data engineering
def calcMACD(data):  #this counts the key statistical indicator for the strategy. MACD in my case
    prices = data['Close']
    indicator = prices.ewm(span=12, adjust=False, min_periods=12).mean() - prices.ewm(span=26, adjust=False, min_periods=26).mean()
    signal = indicator.ewm(span=9, adjust=False, min_periods=9).mean()
    d = indicator - signal
    return d

def ma(data, span):
  mean = []
  for e in range(len(data[span:])):
    mean.append(np.mean(data[e-span:e]))
  return np.array(mean)

def atr(data):
  atr = [0]
  for i in range(1, len(data)):
    pre_tr = [data['High'][i] - data['Low'][i], abs(data['High'][i] - data['Close'][i-1]), abs(data['Low'][i] - data['Close'][i-1])]
    atr.append((atr[-1]*13 + max(pre_tr))/14)
  return np.array(atr)

def bollinger_bands(data, span=20):    ####doesnt fucking work
  bands=[]
  sds = []
  maspan = ma(data, span=span)
  sqrdeviations = [g**2 for g in [data['Close'][i] - np.mean(data['Close'][i-span:i]) for i in range(len(data))]]
  for s in range(len(sqrdeviations)):
    sds.append(np.mean(sqrdeviations[s-20:s])**0.5)
  for e in range(len(maspan)):
    bands.append([maspan[e]**0.5-2*sds[e], maspan[e], maspan[e]**0.5+2*sds[e]])
  return bands

def createdataset_yfbacktest(secu):
  indicators = pd.DataFrame([])
  indicators['open'], indicators['close'], indicators['high'], indicators['low'] = secu.Open[100:], secu.Close[100:], secu.High[100:], secu.Low[100:]
  indicators['macdhist'] = calcMACD(secu)[74:]
  indicators['ma20'], indicators['ma50'] = ma(secu.Close, 20)[80:], ma(secu.Close, 50)[50:]
  indicators['atr'] = atr(secu[100:])
  return indicators

def createdataset(len_of_data, df):
  indicators = pd.DataFrame([])
  indicators['open'], indicators['close'], indicators['high'], indicators['low'] = df['Open'][100:], df['Close'][100:], df['High'][100:], df['Low'][100:]
  indicators['macdhist'] = calcMACD(df)
  indicators['ma20'], indicators['ma50'] = ma(df['Close'], 20)[80:], ma(df['Close'], 50)[50:]
  indicators['atr'] = atr(df)[100:]
  return indicators

def get_trainingdata(indicators, len_of_sample=5, len_of_label=1, scope=1):
  training = []
  labels = []
  training_full = []

  for i in range(0, len(indicators)-len_of_sample-len_of_label+1-scope):
    o = indicators['open'][i:i+len_of_sample]
    c = indicators['close'][i:i+len_of_sample]
    h = indicators['high'][i:i+len_of_sample]
    l = indicators['low'][i:i+len_of_sample]
    macd = indicators['macdhist'][i:i+len_of_sample]
    ma20 = indicators['ma20'][i:i+len_of_sample]
    ma50 = indicators['ma50'][i:i+len_of_sample]
    atr = indicators['atr'][i:i+len_of_sample]
    ins = pd.DataFrame([o,c,h,l,macd,ma20,ma50,atr])                       

    if len_of_label == 1:          
      y = indicators['close'][i+len_of_sample+scope-1]
    else:
      y = indicators['close'][i+len_of_sample+scope-1:i+len_of_sample+scope-1+len_of_label]
    pic = np.reshape(ins.values, (len_of_sample, indicators.shape[1]))

    training.append(pic)
    labels.append(y)
    training_full.append((pic, y))

  training = np.array(training)
  labels = np.array(labels)
  return training, labels

def get_needed_data_tinkoff(TOKEN, FIGI, period=12, len_of_sample : int = 5, len_of_label : int = 1) -> tuple:
    with Client(TOKEN) as client:
        open, close, high, low, volume = [], [], [], [], []

        for candle in client.get_all_candles(
            figi=FIGI,
            from_=now() - timedelta(days=period),
            interval=CandleInterval.CANDLE_INTERVAL_5_MIN
        ):
            open.append(money.Money(candle.open).units + money.Money(candle.open).nano/(10**9))
            close.append(money.Money(candle.close).units + money.Money(candle.close).nano/(10**9))
            high.append(money.Money(candle.high).units + money.Money(candle.high).nano/(10**9))
            low.append(money.Money(candle.low).units + money.Money(candle.low).nano/(10**9))
            volume.append(candle.volume)
        ef = createdataset(len(close), pd.DataFrame({'Open' : open, 'Close' : close, 'High' : high, 'Low' : low})).reset_index(drop=True)
        databits, lbls = get_trainingdata(ef.reset_index(drop=True), len_of_sample=len_of_sample, len_of_label=len_of_label)

    return (ef, databits, lbls)

#pipeline of creating a predictor object: obj = predictor(model_type, lyrs, ...)
#                                         train on a variety of securities
#                                         obj.examine_bias(on a security that the estimator is purposed for)
#                                         ready for making predictions