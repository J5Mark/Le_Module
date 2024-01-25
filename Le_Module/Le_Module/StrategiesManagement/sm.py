from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from PaperworkManagement.papman import *
from PredictorsManagement.Predictors import *
from Utilities.FinUtils import *
from tinkoff.invest import CandleInterval
from StrategiesManagement.DecidingModules import *
from StrategiesManagement.QuantityControllers import *
import math as m

@dataclass
class Squad:
    '''a collection of AIs and algorythms the strategy may be using for assessing the overall trend, predicting upcoming prices and managing risks'''
    Predictors: list[AIPredictor]
    TrendViewers: list[AITrendviewer] | None = None
    RiskManagers: list[Predictor] | None = None

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

    def see_risks(self, databit: Candles, avg_buy_price: float | None=None) -> tuple:
        risky_list_limit = []
        if self.squad.RiskManagers != None:
            risky_list_bool = []
            for rm in self.squad.RiskManagers:
                assessment = rm.predict(databit, avg_buy_price)
                risky_list_bool.append(assessment[1])
                risky_list_limit.append(assessment[0])
            return risky_list_bool, risky_list_limit
        else: return [False], risky_list_limit

class SignalGenerator(ABC):
    '''the core logic module behind the strategy.
    ProvizionWizard is only needed AI predictors or custom risk managers are used'''
    def __init__(self, Decider: DecidingModule, Quantifier: QuantityController, ProvisionWizard: Provizor | None = None):
        self.Wizard = ProvisionWizard
        self.Decider = Decider
        self.Quantifier = Quantifier
    
    def decide(self, databits: DatabitsBatch, avg_buy_price: float | None=None) -> DecisionsBatch:
        if self.Wizard != None:
            trend = self.Wizard.see_trend(databits.for_trendviewers)
            predictions = self.Wizard.predict(databits.for_predictors) 
            risks = self.Wizard.see_risks(databits.for_risk_managers, avg_buy_price)

            market_direction = self.Decider.decide(databits.for_predictors, trend, predictions, risks)
            market_quantity = self.Quantifier.decide_quantity(databits.for_predictors, market_direction, trend, predictions, risks)
            
            stoploss_price = np.mean(risks[0]) if len(risks[0]) != 0 else None
            #takeprofit_price = None #takeprofits are not quite implemented yet
        else:
            market_direction = self.Decider.decide(databits.for_predictors)
            market_quantity = self.Quantifier.decide_quantity(databits.for_predictors, market_direction)

        batch = DecisionsBatch()
        batch.market = Decision(direction=market_direction, amount=market_quantity, type=0, price=-1)
        batch.stop_loss = Decision(direction=False, amount=-1, type=1, price=stoploss_price)
        batch.take_profit = None#Decision(direction=False, amount=-1, type=2, price=takeprofit_price)
        
        return batch

@dataclass
class StrategyParams:
    '''a form of many algorythms the strategy will be using.
      :Squad: is a collection of different purposed models for predicting stock prices, trends and assessing risks.
      :SignalGenerator: is a module containing all of the logic behind the strategy.'''
    SignalGenerator: SignalGenerator 
    TOKEN: str 
    FIGI: str 
    Budget: Money | float = 0.0
    Backtest: bool = True 
    Intervals: tuple(CandleInterval) = (CandleInterval.CANDLE_INTERVAL_3_MIN)

class AutomatedStrategy:
    '''a class for all strategies(its modular)'''

    def __init__(self, params: StrategyParams,  
                 lot_size: int=1,
                 strategy_id: str='', 
                 bridge: Bridge | None=None, 
                 possession: int=0,
                 possession_buy_prices: float=0.0,
                 comission: float=0.05):
        self.params = params
        self.strategy_id = strategy_id
        self.bridge = bridge 
        self.lot_size = lot_size
        self.possession = possession
        self.comission = comission/100
        self.possession_buy_prices = possession_buy_prices
        self.history = {'Budget': [self.params.Budget],
                        'Signals': [None],
                        'Possession': [self.possession],
                        'Buy prices': [possession_buy_prices],
                        'Market sell prices': [None],
                        'Stoploss sell prices': [None],
                        'Takeprofit sell prices': [None],
                        'PnL': [self.params.Budget]}
    
    @property
    @abstractmethod
    def get_params(self) -> StrategyParams:
        return self.params
        
    def _get_signal(self, databit: Candles):
        if self.possession != 0: avg_buy_price = self.possession_buy_prices/self.possession
        else: avg_buy_price = 0
        return self.params.SignalGenerator.decide(databit, avg_buy_price)

    def _real_order(self, signal):
        return self.bridge.post_order(signal)

    def _test_order(self, signal: Decision, price: float | None = None):  
        if self.bridge != None:
            price = self.bridge.current_price()
        else:
            price = price
        if signal.direction: #buy
            self.params.Budget -= (1+self.comission) * price * self.lot_size * signal.amount
            self.possession += self.lot_size * signal.amount
            self.possession_buy_prices += price * signal.amount

        else: #sell
            self.params.Budget += (1-self.comission) * price * self.lot_size * signal.amount
            self.possession -= self.lot_size * signal.amount
            if self.possession != 0:
                self.possession_buy_prices -= signal.amount * self.possession_buy_prices/(self.possession)

    def decide(self, databits: DatabitsBatch, transparency: bool=False):
        
        if self.params.Budget >= databits.for_predictors.Close.values[-1] * self.lot_size:
            signal = self._get_signal(databits)
            
            if signal.market.amount == -1:
                actual_market_amount = self.possession//self.lot_size
            else:
                actual_market_amount = signal.market.amount if self.params.Budget >= signal.market.amount * self.lot_size * databits.for_predictors.Close.values[-1] else m.floor(self.params.Budget / (signal.market.amount * self.lot_size * databits.for_predictors.Close.values[-1]))
            signal.market.amount = actual_market_amount
            
            if not self.params.Backtest:
                market_ord = self._real_order(signal.market)
                stoploss_ord = self._real_order(signal.stop_loss)
                #takeprofit_ord = self._real_order(signal.take_profit)
                

            else:
                self._test_order(Decision(signal.market.direction, actual_market_amount), databits.for_predictors.Close.values[-1])
                    
                self.history['Budget'].append(self.params.Budget)
                self.history['Possession'].append(self.possession)
                self.history['Signals'].append(signal.market)
                self.history['Buy prices' if signal.market.direction else 'Market sell prices'].append(databits.for_predictors.Close.values[-1])
                self.history['Buy prices' if not signal.market.direction else 'Market sell prices'].append(None)
                self.history['PnL'].append(self.params.Budget + self.possession * self.lot_size * databits.for_predictors.Close.values[-1])
                self.history['Stoploss sell prices'].append(None)
                self.history['Takeprofit sell prices'].append(None)

        else:
            self.history['Signals'].append(None)
            self.history['Budget'].append(self.params.Budget)
            self.history['Possession'].append(self.possession)
            self.history['Buy prices'].append(None)
            self.history['Market sell prices'].append(None)
            self.history['PnL'].append(self.params.Budget + self.possession * self.lot_size * databits.for_predictors.Close.values[-1])
            self.history['Stoploss sell prices'].append(None)
            self.history['Takeprofit sell prices'].append(None)

        return transparency*[self._get_signal(databits), databits]

    def clear_history(self):
        self.history = {'Budget': [self.params.Budget],
                        'Signals': [None],
                        'Possession': [self.history['Possession'][0]],
                        'Buy prices': [self.history['Buy prices'][0]],
                        'Market sell prices': [None],
                        'Stoploss sell prices': [None],
                        'Takeprofit sell prices': [None],
                        'PnL': [self.params.Budget]}