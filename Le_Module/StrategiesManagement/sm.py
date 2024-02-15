from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from PaperworkManagement.papman import *
from PredictorsManagement.Predictors import *
from Utilities.FinUtils import *
from tinkoff.invest import CandleInterval
from StrategiesManagement.DecidingModules import *
from StrategiesManagement.QuantityControllers import *
from StrategiesManagement.RiskManagers import *
import math as m

@dataclass
class Squad:
    '''a collection of AIs and algorythms the strategy may be using for assessing the overall trend, predicting upcoming prices and managing risks'''
    Predictors: list[Predictor] | None = None
    TrendViewers: list[AITrendviewer] | None = None
    RiskManagers: list[RiskManager] | None = None
    
@dataclass
class CombinedRMResponses:
    '''dataclass used for organizing what the provizor gets from the squad's risk managers'''
    stoploss_prices: list[float | None]
    risks: list[bool | None]

class Provizor:
    '''a class which manages all of the predictions of an AIPredictors and risk managers squad'''
    def __init__(self, squad: Squad):
        self.squad = squad
    
    def see_trend(self, databit: Candles) -> list | None:
        '''a method intended to assure trading is worth it, databit should definetely be a higher timeframe than the operating data'''
        if self.squad.TrendViewers != None:
            npdatabit = databit.as_nparray()
            tr_predictions = []
            for trviewer in self.squad.TrendViewers:
                tr_predictions.append(trviewer.predict(npdatabit))

            return tr_predictions
        else: return None
    
    def predict(self, databit: Candles) -> list[PredictorResponse] | None:
        npdatabit = databit.as_nparray()
        predictions = []
        if self.squad.Predictors != None:
            for prdctr in self.squad.Predictors:
                predictions.append(prdctr.predict(npdatabit))
                
            return predictions
        else: return None
        
    def see_risks(self, databit: Candles, avg_buy_price: float | None=None) -> CombinedRMResponses:
        stoploss_list = []
        if self.squad.RiskManagers != None:
            risky_list_bool = []
            for rm in self.squad.RiskManagers:
                assessment: RMResponse = rm.predict(databit, avg_buy_price)
                risky_list_bool.append(assessment.risk)
                stoploss_list.append(assessment.stoploss_price)
            return CombinedRMResponses(stoploss_prices=stoploss_list, risks=risky_list_bool)
        else: return CombinedRMResponses(stoploss_prices=None, risks=[False])

class SignalGenerator(ABC):
    '''the core logic module behind the strategy.
    ProvizionWizard is only needed AI predictors or custom risk managers are used'''
    def __init__(self, Decider: DecidingModule, Quantifier: QuantityController, ProvisionWizard: Provizor | None = None, trend_over_prediction_rythm: int=1):
        self.Wizard = ProvisionWizard
        self.Decider = Decider
        self.Quantifier = Quantifier
        self.trend_over_prediction_rythm = trend_over_prediction_rythm
        self.count = trend_over_prediction_rythm
        self.cached_trend = None
    
    def get_signal(self, databits: DatabitsBatch, avg_buy_price: float | None=None) -> DecisionsBatch:
        '''calls every member of the models squad for their predictions'''
        if self.Wizard != None:
            if self.count % self.trend_over_prediction_rythm == 0:
                self.cached_trend = self.Wizard.see_trend(databits.for_trendviewers) if databits.for_trendviewers != None else None
                self.count = 1
            else:
                self.count += 1
            trend = self.cached_trend
            predictions = self.Wizard.predict(databits.for_predictors) if databits.for_predictors != None else None 
            risks = self.Wizard.see_risks(databits.for_risk_managers, avg_buy_price) if databits.for_risk_managers != None else None

            market_direction = self.Decider.decide(databits.for_predictors, trend, predictions, risks.risks)
            market_quantity = self.Quantifier.decide_quantity(databits.for_predictors, market_direction, trend, predictions, risks.risks)
            
            stoploss_price = np.mean(risks.stoploss_prices) if risks.stoploss_prices else None
            #takeprofit_price = None #takeprofits are not quite implemented yet
        else:
            market_direction = self.Decider.decide(databits.for_predictors)
            market_quantity = self.Quantifier.decide_quantity(databits.for_predictors, market_direction)

        batch = DecisionsBatch(market=Decision(direction=market_direction, amount=market_quantity, type=0, price=-1),
                               stop_loss=Decision(direction=False, amount=-1, type=1, price=stoploss_price),
                               take_profit=None#Decision(direction=False, amount=-1, type=2, price=takeprofit_price)
                               )
        
        return batch

@dataclass
class StrategyParams:
    '''a form of many algorythms the strategy will be using.
      :Squad: is a collection of different purposed models for predicting stock prices, trends and assessing risks.
      :SignalGenerator: is a module containing all of the logic behind the strategy.'''
    SignalGenerator: SignalGenerator 
    TOKEN: str 
    FIGI: str 
    Budget: float = 0.0
    Backtest: bool = True 
    Intervals: tuple[CandleInterval] = (CandleInterval.CANDLE_INTERVAL_3_MIN)

class AutomatedStrategy:
    '''a class for all strategies(its modular)'''

    def __init__(self, params: StrategyParams,  
                 lot_size: int=1,
                 strategy_id: str='', 
                 bridge: Bridge | None=None, 
                 possession: int=0,
                 mean_buy_price: float=0,
                 comission: float=0.05):
        self.params = params
        self.strategy_id = strategy_id
        self.bridge = bridge 
        self.lot_size = lot_size
        self.possession = possession
        self.comission = comission/100
        self.mean_buy_price = mean_buy_price
        self.b_prices_list = [mean_buy_price]
        self.history = {'Budget': [self.params.Budget],
                        'Signals': [None],
                        'Possession': [self.possession],
                        'Buy prices': [mean_buy_price],
                        'Market sell prices': [None],
                        'Stoploss sell prices': [None],
                        'Takeprofit sell prices': [None],
                        'PnL': [self.params.Budget]}
    
    @property
    @abstractmethod
    def get_params(self) -> StrategyParams:
        return self.params
        
    def _get_signal(self, databit: Candles):
        avg_buy_price = round(np.mean(self.b_prices_list), 3) if len(self.b_prices_list) != 0 else 0
        return self.params.SignalGenerator.get_signal(databit, avg_buy_price)

    def _real_order(self, signal):
        return self.bridge.post_order(signal)

    def _test_order(self, signal: Decision, price: float | None = None):  
        price = price
        if signal.direction: #buy
            self.params.Budget -= (1+self.comission) * price * self.lot_size * signal.amount
            self.possession += self.lot_size * signal.amount
            self.b_prices_list += [price] * self.lot_size * signal.amount

        elif signal.direction == False: #sell
            self.params.Budget += (1-self.comission) * price * self.lot_size * signal.amount
            self.possession -= self.lot_size * signal.amount
            if self.possession != 0:
                self.b_prices_list = [np.mean(self.b_prices_list[self.lot_size*signal.amount:])] * self.possession
            else:
                self.b_prices_list = []
        else:
            pass #literally shouldn`t do a thing
                
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
                self.history['Buy prices' if signal.market.direction == False else 'Market sell prices'].append(None)
                self.history['PnL'].append(self.params.Budget + self.possession * self.lot_size * databits.for_predictors.Close.values[-1])
                self.history['Stoploss sell prices'].append(None)
                self.history['Takeprofit sell prices'].append(None)

        else:
            self.history['Signals'].append(None)
            self.history['Budget'].append(self.params.Budget)
            self.history['Possession'].append(self.possession)
            self.history['Buy prices'].append(None)
            self.history['Market sell prices'].append(None)
            self.history['PnL'].append(self.possession * self.lot_size * databits.for_predictors.Close.values[-1])
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