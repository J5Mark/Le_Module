from PredictorsManagement.Predictors import Predictor
from Utilities.FinUtils import *

class FallPerceptionRiskManager(Predictor):
    def __init__(self, border_fall: float):
        self.border_fall = border_fall
    
    def predict(self, databit: Candles, avg_buy_price: float) -> bool:
        return databit[-1].Close.values <= avg_buy_price*(1-self.border_fall)