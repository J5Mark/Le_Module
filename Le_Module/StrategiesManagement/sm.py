from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from PaperworkManagement.papman import *
from PredictorsManagement.Predictors import *
from Utilities.FinUtils import *
from Utilities.DataClasses import *
from tinkoff.invest import CandleInterval
from StrategiesManagement.DecidingModules import *
from StrategiesManagement.QuantityControllers import *
from StrategiesManagement.RiskManagers import *
import math as m
import json
from tinkoff.invest.exceptions import *
from tinkoff.invest.services import MarketDataStreamManager
import datetime
import traceback

class PriceDataTypeError(Exception):
    def __init__(self, price_data):
        self.msg = f'Price data (type: {price_data.__class__.__name__}), which is provided is of unfit type. Should be: Candles | Instrument | np.ndarray | list'
        super().__init__(self.msg)

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
        if self.squad.TrendViewers is not None:
            npdatabit = databit.as_nparray()
            tr_predictions = []
            for trviewer in self.squad.TrendViewers:
                tr_predictions.append(trviewer.predict(npdatabit))

            return tr_predictions
        else: return None
    
    def predict(self, databit: Candles) -> list[PredictorResponse] | None:
        npdatabit = databit.as_nparray()
        predictions = []
        if self.squad.Predictors is not None:
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


class AutomatedStrategy:
    '''a class for all strategies(its modular)'''

    def __init__(self, params: StrategyParams,  
                 lot_size: int=1,
                 strategy_id: str='', 
                 bridge: Bridge | None=None, 
                 possession: int=0,
                 mean_buy_price: float=0,
                 commission: float=0.3,
                 transparency: bool=False, 
                 orchestrator=None): #for now in order to not circlejerk imports i`ll leave this typeless
        self.transparency = transparency
        self.orchestrator = orchestrator
        self._params = params
        self.strategy_id = strategy_id
        self.bridge = bridge 
        self.lot_size = lot_size
        self.possession = possession
        self.commission = commission/100
        self.b_prices_list = [mean_buy_price]
        self.txtlog = ''
        self._necessary_initial_period: int | None = None
        self._data_prep: DataPreparationInstructions | None = None
        self.for_predictors: int | None = None
        self.for_risk_managers: int | None = None
        self.for_trendviewers: int | None = None
        self.max_attempts = 1000
        self.distcount = None
        self.distribution_frequency = None
        self.history: BotStats = BotStats(PnL=[self._params.Budget],
                                              Buy_prices=[mean_buy_price if mean_buy_price != 0 else None],
                                              Market_sell=[None], 
                                              Stoploss_sell=[None],
                                              Possession=[self.possession],
                                              Budget=[self._params.Budget],
                                              Signals=[None],
                                              Takeprofit_sell=[None],
                                              commission=commission)
    
    @property
    @abstractmethod
    def get_figi(self) -> str:
        return self._params.FIGI
    
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
        order: Order | None = self.bridge.post_order(decision=signal, figi=self._params.FIGI)
        if isinstance(signal, Decision) and order is not None:
            if signal.direction:
                self._params.Budget -= (1+self.commission/100) * order.price * self.lot_size * order.amount
                self.possession += self.lot_size * order.amount
                self.b_prices_list += [order.price] * self.lot_size * order.amount
            elif signal.direction == False:
                self._params.Budget += (1+self.commission/100) * order.price * self.lot_size * order.amount
                self.possession -= self.lot_size * order.amount
                if self.possession != 0:
                    self.b_prices_list = [np.mean(self.b_prices_list[self.lot_size*order.amount:])] * self.possession
                else:
                    self.b_prices_list = []            
            else:
                print('\n'+f'Order {order.order_id} is not fully or partially executed')
        
    def _test_order(self, signal: Decision, price: float | None = None):  
        price = price
        if signal.direction:#buy
            self._params.Budget -= (1+self.commission) * price * self.lot_size * signal.amount
            self.possession += self.lot_size * signal.amount
            self.b_prices_list += [price] * self.lot_size * signal.amount

        elif signal.direction == False: #sell
            self._params.Budget += (1-self.commission) * price * self.lot_size * signal.amount
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
                actual_market_amount = int(self._params.Budget // (self.lot_size * databits.for_predictors.Close.values[-1]))
                
        elif signal.market.direction == False:
            if signal.market.amount == -1:
                actual_market_amount = self.possession//self.lot_size
        elif signal.market.direction is None:
            actual_market_amount = 0
            self.history.Signals.append(None)
            self.history.Budget.append(self._params.Budget)
            self.history.Possession.append(self.possession)
            self.history.Buy_prices.append(None)
            self.history.Market_sell.append(None)
            self.history.PnL.append(self._params.Budget + self.possession * databits.for_predictors.Close.values[-1])
            self.history.Stoploss_sell.append(None)
            self.history.Takeprofit_sell.append(None)
            
        signal.market.amount = actual_market_amount
                
        if not self._params.Backtest:
                market_ord = self._real_order(signal.market)
                stoploss_ord = self._real_order(signal.stop_loss)
                #takeprofit_ord = self._real_order(signal.take_profit)
                
                direction = signal.market.direction
                statsignal = Decision(direction=direction if direction is None else int(direction), amount=actual_market_amount*self.lot_size, price=signal.market.price)
                self.history.Budget.append(self._params.Budget)
                self.history.Possession.append(self.possession)
                self.history.Signals.append(statsignal)
                self.history.Buy_prices.append(databits.for_predictors.Close.values[-1]) if signal.market.direction else self.history.Market_sell.append(databits.for_predictors.Close.values[-1])
                self.history.Buy_prices.append(None) if signal.market.direction == False or signal.market.direction is None else self.history.Market_sell.append(None)
                self.history.PnL.append(self._params.Budget + self.possession * databits.for_predictors.Close.values[-1])
                self.history.Stoploss_sell.append(None)
                self.history.Takeprofit_sell.append(None)   
                if transparency:
                    databits.for_predictors.Time = None
                    databits.for_risk_managers.Time = None
                    databits.for_trendviewers.Time = None
                    print(f'Databits: \n{databits}\n\nSignals: \n{signal}\n\n')
                return market_ord, stoploss_ord#, takeprofit_ord
                
                
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
            
            if transparency:
                    print(databits, signal)     

    def clear_history(self):
        self.history = BotStats(Budget=[self.history.Budget[0]],
                        Signals=[None],
                        Possession=[self.history.Possession[0]],
                        Buy_prices=[self.history.Buy_prices[0]],
                        Market_sell=[None],
                        Stoploss_sell=[None],
                        Takeprofit_sell=[None],
                        PnL=[self.history['PnL'][0]],
                        commission=self.commission)
        
    def _on_update(self, cached: Candles, new_data: MarketDataResponse | None, orders: list[Order]):
            #somehow trade
            if new_data is not None:
                cached.Close.append(new_data.candle.close.units + 10**(-9)*new_data.candle.close.nano)
                cached.Open.append(new_data.candle.open.units + 10**(-9)*new_data.candle.open.nano)
                cached.High.append(new_data.candle.high.units + 10**(-9)*new_data.candle.high.nano)
                cached.Low.append(new_data.candle.low.units + 10**(-9)*new_data.candle.low.nano)
                cached.Volume.append(new_data.candle.volume)
                cached = self._data_prep(cached)
                
                orders = [v for v in orders if v is not None]
                if len(orders) != 0:
                    canc_1 = [o.order_id for o in orders if o.order_type == 1]
                    canc_2 = [o.order_id for o in orders if o.order_type == 2]
                    self.bridge.cancel_orders(canc_1 + canc_2)

                order = self.decide(databits=DatabitsBatch(for_predictors=cached[-self.for_predictors:],
                                                            for_trendviewers=cached[-self.for_trendviewers:] if self.for_trendviewers else None,
                                                            for_risk_managers=cached[-self.for_risk_managers:] if self.for_risk_managers else None), 
                                    transparency=self.transparency)
                orders += [o for o in order]
                orders = self.bridge.check_orders(orders)
                
                if self.distcount == self.distribution_frequency and self.orchestrator is not None:
                    self.orchestrator(self.strategy_id)
            
            return cached, orders
            
    def trade(self, data_preparation_intructions: DataPreparationInstructions,
              interval: CandleInterval=CandleInterval.CANDLE_INTERVAL_HOUR, 
              necessary_initial_period: int=50,
              decision_frequency: int=1,
              distribution_frequency: int=14, #if a decision is made every hour, after every daytrading session budget would be distributed
              for_predictors: int=2,
              for_risk_managers: int | None=None,
              for_trendviewers: int | None=None,
              waiting_close: bool=True, *args, **kwargs):
        self.distribution_frequency = distribution_frequency
        i = decision_frequency
        self.distcount = distribution_frequency
        self.for_predictors, self.for_risk_managers, self.for_trendviewers = for_predictors, for_risk_managers, for_trendviewers
        self._necessary_initial_period: int = necessary_initial_period
        self._data_prep: DataPreparationInstructions = data_preparation_intructions
        if decision_frequency == 1:
            cached_data = get_data_tinkoff(TOKEN=self._params.TOKEN, FIGI=self._params.FIGI, period=necessary_initial_period, interval=interval)
        else:
            cached_data = get_data_tinkoff(TOKEN=self._params.TOKEN, FIGI=self._params.FIGI, period=necessary_initial_period*3, interval=interval)[::-decision_frequency] #number 3 is here just for the amounts of data to be kinda copious
            cached_data = cached_data[::-1]
        cached_data = data_preparation_intructions(cached_data)
        orders: list[Order] | None = []
        with open(f'trading_log_{self.strategy_id}.txt', 'w+') as file:
            file.write(f'Trading started. time : {datetime.datetime.now()}')
            file.flush()
            
            with Client(token=self._params.TOKEN, app_name=self.strategy_id) as client:
                trading_status = client.market_data.get_trading_status(figi=self._params.FIGI)
                if not trading_status.market_order_available_flag:
                    file.write('\n'+f'Market trading is unavailable right now for figi {self._params.FIGI}')
                    file.flush()
                datastream: MarketDataStreamManager = client.create_market_data_stream()
                subscription = datastream.candles.waiting_close(enabled=waiting_close).subscribe([CandleInstrument(figi=self._params.FIGI, interval=interval)])
                file.write('\n'+f'Subscribed to datastream. Interval: {interval}'+'\n')
                file.flush()
                for e in range(self.max_attempts):
                    try:
                        if subscription is None:
                            datastream: MarketDataStreamManager = client.create_market_data_stream()
                            subscription = datastream.candles.waiting_close(enabled=waiting_close).subscribe([CandleInstrument(figi=self._params.FIGI, interval=interval)])
                            file.write(f'\nError handled successfully, trading continues. Time: {datetime.datetime.now()}\n')
                        for new_data in datastream:
                            if new_data.candle:
                                file.write('\n'+f'{self.strategy_id} recieved market data: {new_data.candle.close} (showing only candle close). time : {datetime.datetime.now()}\n\n')
                                file.flush()
                            else:
                                file.write('.')
                                file.flush()
                            if new_data.candle:
                                if self.distcount == distribution_frequency and self.orchestrator is not None:
                                    dist = self.orchestrator(self.strategy_id)
                                    if dist:
                                        prev_budget = self._params.Budget
                                        self.set_budget(dist[self.strategy_id])
                                        file.write(f'Budget redistributed by bot`s orchestrator(was: {prev_budget}, now: {self._params.Budget}. Time: {datetime.datetime.now()}')
                                        file.flush()
                                        self.distcount = 1
                                if i == decision_frequency:
                                    cached_data, orders = self._on_update(cached_data, new_data, orders)
                                    i = 0
                                    with open(f'trading_{self.strategy_id}.json', 'w+') as jsonfile:
                                        dictforjason = {k : self.history.__dict__[k] for k in list(self.history.__dict__.keys())}
                                        json.dump(obj=dictforjason, fp=jsonfile, indent=4, separators=(',', ': '), cls=EnhancedJSONEncoder)                                    
                                    file.write('\n'+f'#### decision made : \n------------\n{self.history.Signals[-1].direction}\n{self.history.Signals[-1].amount}\n{self.history.Signals[-1].type}\n------------\n\n')
                                    file.flush()      
                                i += 1
                                self.distcount += 1
                            if new_data.trading_status and new_data.trading_status.market_order_available_flag:
                                file.write('\n'+f'Trading is limited by broker or joever, current status: {new_data.trading_status}\n')
                                file.flush()
                                break
                            
                    except RequestError as err:
                        if e < self.max_attempts:
                            file.write('\n'+f'{err} encountered, handling(number of attempt: {e+1} / {self.max_attempts})... time : {datetime.datetime.now()}')
                            file.flush()
                            print(f'at time : {datetime.datetime.now()} \nencountered error : {err}\nhandling(number of attempt: {e+1} / {self.max_attempts})...')
                            for each in self.history.__dict__:
                                match self.history.__dict__[each]:
                                    case list():
                                        self.history.__dict__[each][-1] = self.history.__dict__[each][-2]
                                    case _:
                                        self.history.__dict__[each] = self.history.__dict__[each]
                                        
                            self._params.Budget = self.history.Budget[-1]
                            datastream.unsubscribe(MarketDataRequest(subscribe_candles_request=subscription))
                            subscription = None
                            datastream.stop()
                            continue
                        elif e >= self.max_attempts:
                            file.write('\nError handling attempts exhausted, stopping trading')
                            datastream.unsubscribe(MarketDataRequest(subscribe_candles_request=subscription))
                            subscription = None
                            datastream.stop() 
                    except KeyboardInterrupt:
                        file.write('\n'+f'trading stopped by user request. time : {datetime.datetime.now()}')
                        file.flush()
                        datastream.unsubscribe(MarketDataRequest(subscribe_candles_request=subscription))
                        subscription = None
                        datastream.stop()
                        break
                    except InvestError as err:
                        file.write('\n'+f'Trading stopped due to error: {err}')
                        file.flush()
                        datastream.unsubscribe(MarketDataRequest(subscribe_candles_request=subscription)) 
                        subscription = None
                        datastream.stop()
                        break
                    except Exception as err:
                        print(traceback.format_exc())
                        file.write('\n'+f'{err} encountered, handling(number of attempt: {e+1} / {self.max_attempts})... time : {datetime.datetime.now()}')
                        file.flush()
                        print(f'at time : {datetime.datetime.now()} \nencountered an Unknown Error \nhandling(number of attempt: {e+1} / {self.max_attempts})...')
                        datastream.unsubscribe(MarketDataRequest(subscribe_candles_request=subscription))
                        subscription = None
                        datastream.stop()
                        if e <= self.max_attempts:
                            datastream: MarketDataStreamManager = client.create_market_data_stream()
                            datastream.candles.waiting_close(enabled=waiting_close).subscribe([CandleInstrument(figi=self._params.FIGI, interval=interval)])
                            print('error handled successfully')
                    finally:
                        datastream.unsubscribe(MarketDataRequest(subscribe_candles_request=subscription))
                        subscription = None
                        datastream.stop()
            
class BotStats:        
    def __init__(self, PnL: list=[], Buy_prices: list=[], 
                Market_sell: list=[], Stoploss_sell: list=[], 
                Possession: list=[], Budget: list=[], 
                Signals: list[Decision | None]=[], Takeprofit_sell: list=[],
                commission: float=0.3):
        self.commission = commission/100
    
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
        self.rise_per_1_up = None
        self.rise_per_1_down = None 
            
    def calc_deals_stats(self):
        current_deal = []
        profits = []
        losses = []
        for i in range(len(self.Buy_prices)):
            buy_price = self.Buy_prices[i]
            market_sell_price = self.Market_sell[i]
            amount = 0 if self.Signals[i] is None else self.Signals[i].amount
            if self.Signals[i] is not None:
                if not all([buy_price is None, market_sell_price is None]):
                    if buy_price is not None:
                        current_deal.append(buy_price*(1+self.commission)*amount)
                    elif market_sell_price is not None:
                        ret = market_sell_price*(1-self.commission) - np.mean(current_deal)
                        if ret > 0:
                            profits.append(ret*amount)
                        elif ret < 0:
                            losses.append(ret*amount)
            else:
                continue
                
        self.mean_profit = np.mean(profits)
        self.mean_loss = np.mean(losses)
        self.winning_rate = len(profits)/(len(profits) + len(losses)) if len(profits) + len(losses) != 0 else None
        
    def calc_stats(self, risk_free_rate):        
        self.profit = (self.PnL[-1] - self.Budget[0])
        money_used = (self.Budget[0] - min(self.Budget))
        self.efficiency = (self.profit / money_used).round(4)
        self.mdd, self.pdd, self.mdd_start, self.mdd_end = all_about_drawdowns(np.array(self.PnL)-self.Budget[0])
        self.sharpe_ratio = sharpe_ratio(self.PnL, riskfree=risk_free_rate)   
        
    def calc_extrusion_rate(self, price_data: Candles | Instrument | list | np.ndarray):    
        match price_data:
            case list():
                d = np.array(price_data)
            case np.ndarray() if price_data.ndim == 1:
                d = price_data
            case np.ndarray() if price_data.ndim != 1:
                raise WrongDimsNdArrayError(correct_dims=1, provided_dims=price_data.ndim)
            case Candles():
                d = price_data.Close.values
            case Instrument():
                d = price_data.values
            case _:
                raise PriceDataTypeError(price_data=price_data)
        
        diflen = len(d) - len(self.PnL)
        rise = 0; rprofit = 0
        fall = 0; fprofit = 0
        risemask = [True if d[i] > d[i-1] else False for i in range(diflen+1, len(d))]
        pnlriseperiods = []
        pnlfallperiods = []
        pnlrise = []
        pnlfall = []
        pricerise = []
        pricefall = []
        priceriseperiods = []
        pricefallperiods = []
        for u in range(1, len(risemask)):
            if risemask[u]: 
                if self.PnL[u-1] != self.PnL[u]:
                    pnlrise.append(self.PnL[u-1])
                    pnlrise.append(self.PnL[u]) 
                if pnlfall != []: pnlfallperiods.append(pnlfall)
                pnlfall = []
                
                pricerise.append(d[diflen+u]) 
                if pricefall != []: pricefallperiods.append(pricefall)
                pricefall = []
            else: 
                if self.PnL[u-1] != self.PnL[u]:
                    pnlfall.append(self.PnL[u-1])
                    pnlfall.append(self.PnL[u]) 
                if pnlrise != []: pnlriseperiods.append(pnlrise)
                pnlrise = []
                
                pricefall.append(d[diflen+u]) 
                if pricerise != []: priceriseperiods.append(pricerise)
                pricerise = []
        for i, pnlr in enumerate(pnlriseperiods):
            rise += priceriseperiods[i][-1] - priceriseperiods[i][0]
            rprofit += pnlr[-1] - pnlr[0]
        for i, pnlf in enumerate(pnlfallperiods):
            fall += pricefallperiods[i][0] - pricefallperiods[i][-1]
            fprofit += pnlf[-1] - pnlf[0]
        
        self.rise_per_1_up = rise / rprofit
        self.rise_per_1_down = fall / fprofit            #have to check this really hard
        
    def display_stats(self):
        profit = (self.PnL[-1] - self.Budget[0])
        money_used = (self.Budget - min(self.Budget))
        d = {'profit':profit, 'efficiency':self.efficiency, 
                     'mdd(%)':self.mdd, 'pdd':self.pdd,
                     'mean loss':self.mean_loss, 'mean profit':self.mean_profit,  
                     'rise per 1 down':self.rise_per_1_down, 'rise per 1 up':self.rise_per_1_up,
                     'sharpe ratio':self.sharpe_ratio, 
                     'winning rate':self.winning_rate}  
        for each in list(d.keys()):
            if each is not None:
                print(f'{each} : {d[each]}')
            else: print(f'{each} is not calculated')
        figure, axis = plt.subplots(2, 2, figsize=(15, 15))
        axis[0, 0].plot(self.PnL)
        if all([self.mdd_start, self.mdd_end]):
            axis[0, 0].plot([self.mdd_start, self.mdd_end], [self.PnL[self.mdd_start], self.PnL[self.mdd_end]], 'r.')
        axis[0, 1].plot(self.Budget)
        axis[1, 0].plot(self.Buy_prices, 'g')
        axis[1, 0].plot(self.Market_sell, 'r')
        axis[1, 1].plot(self.Possession)
        for e in axis:
            for i in e:
                i.grid(True)
        plt.show()
            
