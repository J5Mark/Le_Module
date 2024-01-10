from dataclasses import dataclass
from Utilities.FinUtils import *
from tinkoff.invest import CandleInterval
import keras
from StrategiesManagement.sm import SignalGenerator
from PredictorsManagement.Predictors import AIPredictor, AITrendviewer, Predictor

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
            
@dataclass
class PredictorParams:
    '''dataclass for storing all upcoming AI predictor parameters. 
     :structure: is a list of all layers of a future predictor'''
    loss: keras.losses.Loss
    optimizer: keras.optimizers.Optimizer | str
    structure: list[keras.layers.Layer]
    scope: int=1
    input_len: int=5   
    output_len: int=1 

    epochs: int=300
    training_verbose: bool=False
    callbacks: list | None = None
    predictable: str = 'close'
    metrics: list[str] | None = None