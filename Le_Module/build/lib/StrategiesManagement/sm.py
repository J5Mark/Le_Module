from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from PaperworkManagement.papman import *
from PredictorsManagement.Predictors import *
from Utilities.FinUtils import *
from tinkoff.invest import CandleInterval, OrderState
from StrategiesManagement.DecidingModules import *
from StrategiesManagement.QuantityControllers import *
from StrategiesManagement.RiskManagers import *
import math as m
import json
import asyncio

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
        
    def see_risks(self, databit: Candles, budget: float | None=None, avg_buy_price: float | None=None, last_buy_price: float | None=None, first_buy_price: float | None=None, predictions: list[PredictorResponse] | None=None) -> CombinedRMResponses:
        stoploss_list = []
        if self.squad.RiskManagers is not None:
            risky_list_bool = []
            for rm in self.squad.RiskManagers:
                assessment: RMResponse = rm.predict(databit, budget, avg_buy_price, last_buy_price, first_buy_price, predictions)
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
    
    def get_signal(self, databits: DatabitsBatch, budget: float | None=None, avg_buy_price: float | None=None, last_buy_price: float | None=None, first_buy_price: float | None=None) -> DecisionsBatch:
        '''calls every member of the models squad for their predictions'''
        stoploss = None
        if self.Wizard is not None:
            if self.count % self.trend_over_prediction_rythm == 0:
                self.cached_trend = self.Wizard.see_trend(databits.for_trendviewers) if databits.for_trendviewers != None else None
                self.count = 1
            else:
                self.count += 1
            trend = self.cached_trend
            predictions: list[PredictorResponse] = self.Wizard.predict(databits.for_predictors) if databits.for_predictors != None else None 
            risks: CombinedRMResponses = self.Wizard.see_risks(databits.for_risk_managers, budget, avg_buy_price, last_buy_price, first_buy_price, predictions) if databits.for_risk_managers != None else CombinedRMResponses(stoploss_prices=[], risks=[False])

            market_direction = self.Decider.decide(databits.for_predictors, trend, predictions, risks.risks if risks else None)
            market_quantity = self.Quantifier.decide_quantity(databits.for_predictors, market_direction, trend, predictions, risks.risks)
            
            stoploss_price = np.mean(risks.stoploss_prices) if risks.stoploss_prices else None
            #takeprofit_price = None #takeprofits are not quite implemented yet
            
            stoploss = Decision(direction=False, amount=-1, type=1, price=stoploss_price)
        else:
            market_direction = self.Decider.decide(databits.for_predictors)
            market_quantity = self.Quantifier.decide_quantity(databits.for_predictors, market_direction)

        batch = DecisionsBatch(market=Decision(direction=market_direction, amount=market_quantity, type=0, price=-1),
                               stop_loss=stoploss,
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
                 comission: float=0.3):
        self._params = params
        self.strategy_id = strategy_id
        self.bridge = bridge 
        self.lot_size = lot_size
        self.possession = possession
        self.comission = comission/100
        self.b_prices_list = [mean_buy_price]
        self.history: BotStats = BotStats(PnL=[self._params.Budget],
                                              Buy_prices=[mean_buy_price if mean_buy_price != 0 else None],
                                              Market_sell=[None], 
                                              Stoploss_sell=[None],
                                              Possession=[self.possession],
                                              Budget=[self._params.Budget],
                                              Signals=[None],
                                              Takeprofit_sell=[None])
    
    @property
    @abstractmethod
    def get_params(self) -> StrategyParams:
        return self._params
    
    @abstractmethod
    def set_budget(self, budget: float) -> None:
        self._params.Budget = budget
        
    def _get_signal(self, databit: Candles):
        avg_buy_price = round(np.mean(self.b_prices_list), 3) if len(self.b_prices_list) != 0 else 0
        last_buy_price = self.b_prices_list[-1] if len(self.b_prices_list) != 0 else 0
        first_buy_price = self.b_prices_list[0] if len(self.b_prices_list) != 0 else 0
        return self._params.SignalGenerator.get_signal(databit, self._params.Budget, avg_buy_price, last_buy_price, first_buy_price)

    def _real_order(self, signal: Decision):
        order: Order = self.bridge.post_order(signal)
        orderstate: OrderState = self.bridge.check_orders([order.order_id])[0]
        if orderstate.execution_report_status in [1, 5]:
            if signal.direction:#buy
                self._params.Budget -= (1+orderstate.executed_commission) * order.price * self.lot_size * orderstate.lots_executed
                self.possession += self.lot_size * orderstate.lots_executed
                self.b_prices_list += [order.price] * self.lot_size * orderstate.lots_executed

            elif signal.direction == False: #sell
                self._params.Budget += (1+orderstate.executed_commission) * order.price * self.lot_size * orderstate.lots_executed
                self.possession -= self.lot_size * orderstate.lots_executed
                if self.possession != 0:
                    self.b_prices_list = [np.mean(self.b_prices_list[self.lot_size*orderstate.lots_executed:])] * self.possession
                else:
                    self.b_prices_list = []            
        else: 
            print(f'Order {order.order_id} is not fully or partially executed')

        
    def _test_order(self, signal: Decision, price: float | None = None):  
        price = price
        if signal.direction:#buy
            self._params.Budget -= (1+self.comission) * price * self.lot_size * signal.amount
            self.possession += self.lot_size * signal.amount
            self.b_prices_list += [price] * self.lot_size * signal.amount

        elif signal.direction == False: #sell
            self._params.Budget += (1-self.comission) * price * self.lot_size * signal.amount
            self.possession -= self.lot_size * signal.amount
            if self.possession != 0:
                self.b_prices_list = [np.mean(self.b_prices_list[self.lot_size*signal.amount:])] * self.possession
            else:
                self.b_prices_list = []
                
    def decide(self, databits: DatabitsBatch, transparency: bool=False):
        signal = self._get_signal(databits)
        
        if signal.market.direction:
            if self._params.Budget >= databits.for_predictors.Close.values[-1] * self.lot_size * signal.market.amount:
                actual_market_amount = signal.market.amount
            else:
                actual_market_amount = m.floor(self._params.Budget / (signal.market.amount * self.lot_size * databits.for_predictors.Close.values[-1]))
                
        elif signal.market.direction == False:
            if signal.market.amount == -1:
                    actual_market_amount = self.possession//self.lot_size
        else:
            self.history.Signals.append(None)
            self.history.Budget.append(self._params.Budget)
            self.history.Possession.append(self.possession)
            self.history.Buy_prices.append(None)
            self.history.Market_sell.append(None)
            self.history.PnL.append(self._params.Budget + self.possession * databits.for_predictors.Close.values[-1])
            self.history.Stoploss_sell.append(None)
            self.history.Takeprofit_sell.append(None)
            
            return transparency*[signal, databits]
        
        signal.market.amount = actual_market_amount
        
        if not self._params.Backtest:
                market_ord = self._real_order(signal.market)
                stoploss_ord = self._real_order(signal.stop_loss)
                #takeprofit_ord = self._real_order(signal.take_profit)
                
                
        else:
            self._test_order(Decision(signal.market.direction, actual_market_amount), databits.for_predictors.Close.values[-1])
            
            statsignal = Decision(direction=signal.market.direction, amount=signal.market.amount*self.lot_size, price=signal.market.price)
            self.history.Budget.append(self._params.Budget)
            self.history.Possession.append(self.possession)
            self.history.Signals.append(statsignal)
            self.history.Buy_prices.append(databits.for_predictors.Close.values[-1]) if signal.market.direction else self.history.Market_sell.append(databits.for_predictors.Close.values[-1])
            self.history.Buy_prices.append(None) if signal.market.direction == False or signal.market.direction is None else self.history.Market_sell.append(None)
            self.history.PnL.append(self._params.Budget + self.possession * databits.for_predictors.Close.values[-1])
            self.history.Stoploss_sell.append(None)
            self.history.Takeprofit_sell.append(None)
        
        return transparency*[signal, databits]
        

    def clear_history(self):
        self.history = BotStats(Budget=[self.history.Budget[0]],
                        Signals=[None],
                        Possession=[self.history.Possession[0]],
                        Buy_prices=[self.history.Buy_prices[0]],
                        Market_sell=[None],
                        Stoploss_sell=[None],
                        Takeprofit_sell=[None],
                        PnL=[self.history['PnL'][0]])
        
    async def trade(self, data_preparation_intructions: DataPreparationInstructions,
              seconds_to_sleep: int=15*60, 
              interval: CandleInterval=CandleInterval.CANDLE_INTERVAL_2_HOUR, 
              days_of_data_to_fetch: float=1,
              transparency: bool=True,
              for_predictors: int=3,
              for_risk_managers: int=3,
              for_trendviewer: int=3):
                
        tr = True
        print('to stop trading press: Ctrl+C')
        
        while tr:
            try:
                beginning = time.time()
                
                db_ = get_data_tinkoff(self._params.TOKEN, FIGI=self._params.FIGI, period=days_of_data_to_fetch, interval=interval)
                db_ = data_preparation_intructions(db_) 
                db = DatabitsBatch(for_predictors=db_[-for_predictors:], for_risk_managers=db_[-for_risk_managers:], for_trendviewers=db_[-for_trendviewer:])
                signal: list[DecisionsBatch] = self.decide(db, transparency=True) if len(db_.Close.values) != 0 else None
                if transparency:
                    print(f'{self.strategy_id} : {signal[0].market.direction if signal is not None else None} | price: {db_.Close.values[-1]if len(db_.Close.values) != 0 else None}')
                
                    print(f'\nstep done at time: {time.time()}\n\n\n')
                
                end = time.time()
                
                time.sleep(seconds_to_sleep - (end - beginning))
                
            except KeyboardInterrupt:
                with open(f'trading_{self.strategy_id}.json', 'w+') as file:
                    dict_ = {k : self.history.__dict__[k] for k in list(self.history.__dict__.keys())}
                    json.dump(dict_, file)  
                    
                profit = (self.history.PnL[-1] - self._params.Budget)
                money_used = (self._params.Budget - min(self.history.Budget))
                print(f'total profit / money used: {(profit / money_used).round(4)}')
                figure, axis = plt.subplots(2, 2, figsize=(15, 15))
                axis[0, 0].plot(self.history.PnL)
                axis[0, 1].plot(self.history.Budget)
                axis[1, 0].plot(self.history.Buy_prices, 'g')
                axis[1, 0].plot(self.history.Market_sell, 'r')
                axis[1, 1].plot(self.history.Possession)
                for e in axis:
                    for i in e:
                        i.grid(True)
                plt.show()
                
                tr=False
            
            
class BudgetDistributor(ABC):
    def __init__(self, TOKEN: str, interval: CandleInterval):
        self.token = TOKEN
        self.interval = interval
        
    def _get_data_make_indicators(self, bots: list[AutomatedStrategy]) -> dict[AutomatedStrategy : Candles]:
        for_predictions: dict[AutomatedStrategy : Candles]
        for bot in bots:
            d = get_data_tinkoff(TOKEN=self.token, FIGI=bot.get_params.FIGI, period=30, interval=self.interval)
            for_predictions[bot] = createdataset_tinkoff(d)
                
        return for_predictions
    
    @abstractmethod
    def __call__(self, bots: list[AutomatedStrategy], budget: float) -> dict[AutomatedStrategy : float]:
        pass
            
            
class BotsOrganiser:
    def __init__(self, bots: list[AutomatedStrategy], budget: float):
        self.bots = bots
        self.budget = budget
        
    def distrib_budget(self, distributor: BudgetDistributor):
        distr: dict[AutomatedStrategy : float] = distributor(self.bots, self.budget)
        for each in self.bots:
            each.set_budget(self.budget * distr[each])
            
class BotStats:        
    def calc_deals_stats(self):
        current_deal = []
        profits = []
        losses = []
        for i in range(len(self.Buy_prices)):
            thig1 = self.Buy_prices[i]
            thig2 = self.Market_sell[i]
            if not all([thig1 is None, thig2 is None]):
                if thig1 is not None:
                    current_deal.append(thig1)
                elif thig2 is not None:
                    ret = thig2 - np.mean(current_deal)
                    if ret > 0:
                        profits.append(ret*self.Signals[i].amount)
                    elif ret < 0:
                        losses.append(ret*self.Signals[i].amount)
            else:
                continue
                
        self.mean_profit = np.mean(profits)
        self.mean_loss = np.mean(losses)
        self.winning_rate = len(profits)/(len(profits) + len(losses)) 
        
    def calc_stats(self):
        self.profit = (self.PnL[-1] - self.Budget[0])
        money_used = (self.Budget[0] - min(self.Budget))
        self.efficiency = (self.profit / money_used).round(4)
        self.mdd, self.pdd, self.mdd_start, self.mdd_end = all_about_drawdowns(np.array(self.PnL)-self.Budget[0])
        self.sharpe_ratio = sharpe_ratio(self.PnL)        
    
    def __init__(self, PnL: list=[], Buy_prices: list=[], 
                    Market_sell: list=[], Stoploss_sell: list=[], 
                    Possession: list=[], Budget: list=[], 
                    Signals: list[Decision | None]=[], Takeprofit_sell: list=[],
                    amounts: list[int]=[]):
            self.bots_amounts = amounts
        
            self.PnL = PnL
            self.Buy_prices = Buy_prices[1:]
            self.Market_sell = Market_sell[1:]
            self.Stoploss_sell = Stoploss_sell[1:]
            self.Possession = Possession
            self.Budget = Budget
            self.Signals = Signals
            self.Takeprofit_sell = Takeprofit_sell[1:]
            
            self.mean_profit = None
            self.mean_loss = None
            self.winning_rate = None
            
            self.efficiency = None
            self.profit = None
            self.mdd = None
            self.pdd = None
            self.sharpe_ratio = None 
            self.mdd_start = None
            self.mdd_end = None