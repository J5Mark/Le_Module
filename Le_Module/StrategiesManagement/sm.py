from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from PaperworkManagement.Bridges import *
from PredictorsManagement.Predictors import *
from Utilities.FinUtils import *
from tinkoff.invest import CandleInterval
from StrategiesManagement.DecidingModules import *
from StrategiesManagement.QuantityControllers import *
import math as m
import random as rd

@dataclass
class Squad:
    '''a collection of AIs and algorythms the strategy may be using for assessing the overall trend, predicting upcoming prices and managing risks'''
    Predictors: list[AIPredictor]
    TrendViewers: list[AITrendviewer] | None = None
    RiskManagers: list[Predictor] | None = None

@dataclass
class Decision:
    '''dataclass for trading decisions.
      :direction: True - buy, False - sell
      :amount: how many lots to buy/sell'''
    direction: bool
    amount: int
    # These ones are really important for real trading, not for backtesting:
    type: int=0
    price: float=-1

class Provizor:
    '''a class which manages all of the predictions of an AIPredictors and risk managers squad'''
    def __init__(self, squad: Squad):
        self.squad = squad
    
    def see_trend(self, databit: Candles) -> list:
        '''a method intended to assure trading is worth it, databit should definetely be a higher timeframe than the operating data'''
        if self.squad.TrendViewers != None:
            npdatabit = databit.as_nparray()
            tr_predictions = []
            for trviewer in self.squad.TrendViewers:
                tr_predictions.append(trviewer.predict(npdatabit))
        
            return tr_predictions
        else: return None
    
    def predict(self, databit: Candles) -> np.ndarray:
        npdatabit = databit.as_nparray()
        predictions = []
        for prdctr in self.squad.Predictors:
            predictions.append(prdctr.predict(npdatabit))

        return predictions

    def see_risks(self, databit: Candles, avg_buy_price: float | None=None) -> list[bool]:
        if self.squad.RiskManagers != None:
            risky_list = []
            for rm in self.squad.RiskManagers:
                risky_list.append(rm.predict(databit, avg_buy_price))

            return risky_list
        else: return None 

class SignalGenerator(ABC):
    '''the core logic module behind the strategy.
    ProvizionWizard is only needed AI predictors or custom risk managers are used'''
    def __init__(self, Decider: DecidingModule, Quantifier: QuantityController, ProvisionWizard: Provizor | None = None):
        self.Wizard = ProvisionWizard
        self.Decider = Decider
        self.Quantifier = Quantifier
    
    def decide(self, databit: Candles, avg_buy_price: float | None=None) -> Decision:
        if self.Wizard != None:
            trend = self.Wizard.see_trend(databit)
            predictions = self.Wizard.predict(databit)
            risks = self.Wizard.see_risks(databit, avg_buy_price)

            direction = self.Decider.decide(databit, trend, predictions, risks)
            quantity = self.Quantifier.decide_quantity(databit, direction, trend, predictions, risks)
        else:
            direction = self.Decider.decide(databit)
            quantity = self.Quantifier.decide_quantity(databit)

        return Decision(direction=direction, amount=quantity)

@dataclass
class StrategyParams:
    '''a form of many algorythms the strategy will be using.
      :Squad: is a collection of different purposed models for predicting stock prices, trends and assessing risks.
      :SignalGenerator: is a module containing all of the logic behind the strategy.'''
    
    def __init__(self, SignalGenerator: SignalGenerator, 
                 TOKEN: str, 
                 FIGI: str, 
                 Budget: Money | float = 0.0,
                 Backtest: bool = True, 
                 Intervals: list[CandleInterval] = [CandleInterval.CANDLE_INTERVAL_5_MIN]):
        self.SignalGenerator = SignalGenerator
        self.TOKEN = TOKEN
        self.FIGI = FIGI
        self.Budget = Budget
        self.Backtest = Backtest
        self.Intervals = Intervals
        if self.Backtest:
            self.Intervals = f'tinkoff CandleInterval code: {str(*self.Intervals)}'


class AutomatedStrategy:
    '''a class for all strategies(its modular)'''

    def __init__(self, params: StrategyParams, disable_default_risk_management: bool=False, strategy_id: str='', bridge: PaperBridge | None=None):
        self.params = params
        self.disable_default_risk_management = disable_default_risk_management
        self.strategy_id = strategy_id
        self.bridge = bridge 
        self.lot_size = 1
        self.possession = 0
        self.possession_buy_prices = 0
        self.history = {'Budget': [self.params.Budget],
                        'Signals': [None],
                        'Possession': [0],
                        'Buy prices': [None],
                        'Sell prices': [None],
                        'PnL': [self.params.Budget]}
    
    @property
    @abstractmethod
    def get_params(self) -> StrategyParams:
        return self.params
        
    def _get_signal(self, databit: Candles):
        if self.possession != 0: avg_buy_price = self.possession_buy_prices/self.possession
        else: avg_buy_price = 0
        return self.params.SignalGenerator.decide(databit, avg_buy_price)

    def default_RM(self, candles: Candles) -> bool:
        '''outputs True if action needed'''
        if not self.disable_default_risk_management:
            atr = candles.ATR.values[-1]
            price = candles.Close.values[-1]
            return self.possession_buy_prices/self.possession < price - atr
        else:
            return False

    def _real_order(self, signal): #not really well implemented yet
        self.bridge.post_order(signal)

    def _test_order(self, signal: Decision, price: float | None = None):  
        if self.bridge != None:
            price = self.bridge.current_price()
        else:
            price = price
        if signal.direction: #buy
            self.params.Budget -= 1.0005 * price * self.lot_size * signal.amount
            self.possession += self.lot_size * signal.amount
            self.possession_buy_prices += price * signal.amount

        else: #sell
            self.params.Budget += 0.9995 * price * self.lot_size * signal.amount
            self.possession -= self.lot_size * signal.amount
            if self.possession != 0:
                self.possession_buy_prices -= signal.amount * self.possession_buy_prices/(self.possession)

    def decide(self, databit: Candles):
        
        if self.params.Budget >= databit.Close.values[-1] * self.lot_size:
            signal = self._get_signal(databit)
            
            if signal.amount == -1:
                actual_amount = self.possession/self.lot_size
            else:
                actual_amount = signal.amount if self.params.Budget >= signal.amount * self.lot_size * databit.Close.values[-1] else m.floor(self.params.Budget / (signal.amount * self.lot_size * databit.Close.values[-1]))

            if not self.disable_default_risk_management and databit.ATR == None:
                    cndls = get_data_tinkoff(self.params.TOKEN, self.params.FIGI, period=1, interval=self.params.Interval)
                    cndls.ATR = calcATR(cndls)
            else:
                cndls = databit

            if not self.params.Backtest:
                if not self.default_RM(cndls):
                    ord = self._real_order(Decision(signal.direction, actual_amount))
                else:
                    ord = self._real_order(Decision(False, self.possession/self.lot_size))



            else:
                if not self.default_RM(cndls):
                    self._test_order(Decision(signal.direction, actual_amount), cndls.Close.values[-1])
                else:
                    self._test_order(Decision(False, self.possession/self.lot_size), cndls.Close.values[-1])
                    
                self.history['Budget'].append(self.params.Budget)
                self.history['Possession'].append(self.possession)
                self.history['Signals'].append(signal)
                self.history['Buy prices' if signal.direction else 'Sell prices'].append(databit.Close.values[-1])
                self.history['Buy prices' if not signal.direction else 'Sell prices'].append(None)
                self.history['PnL'].append(self.params.Budget + self.possession * self.lot_size * databit.Close.values[-1])

        else:
            self.history['Signals'].append(None)
            self.history['Budget'].append(self.params.Budget)
            self.history['Possession'].append(self.possession)
            self.history['Buy prices'].append(None)
            self.history['Sell prices'].append(None)
            self.history['PnL'].append(self.params.Budget + self.possession * self.lot_size * databit.Close.values[-1])

    def clear_history(self):
        self.history = {'Budget': [self.params.Budget],
                        'Signals': [None],
                        'Possession': [0],
                        'Buy prices': [None],
                        'Sell prices': [None],
                        'PnL': [self.params.Budget]}