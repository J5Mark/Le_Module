from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from tinkoff.invest import CandleInterval
from StrategiesManagement.DecidingModules import *
from StrategiesManagement.QuantityControllers import *
from StrategiesManagement.RiskManagers import *
from StrategiesManagement.sm import *


class BudgetDistributor(ABC):
    def __init__(self, TOKEN: str, interval: CandleInterval, dataprep_instructions: DataPreparationInstructions):
        self.token = TOKEN
        self.interval = interval
        self.dataprep = dataprep_instructions
        
    def _get_data_make_indicators(self, bots: list[AutomatedStrategy], period_in_days: float, interval: CandleInterval) -> dict[AutomatedStrategy : Candles]:
        prepared_data: dict[AutomatedStrategy : Candles]
        for bot in bots:
            d = get_data_tinkoff(self.token, bot.get_figi(), period_in_days, interval)
            prepared_data[bot] = self.dataprep(d)
                
        return prepared_data
    
    @abstractmethod
    def __call__(self, bots: list[AutomatedStrategy], budget: float) -> dict[AutomatedStrategy : float]:
        pass  #should be defined by the user
            
class Orchestrator:
    def __init__(self, bots: list[AutomatedStrategy], budget: float, budget_distributor: BudgetDistributor):
        self.bots = [b for b in bots if b.orchestrator is not None]
        self.budget = budget
        self.distributor = budget_distributor
        
    def distrib_budget(self, distributor: BudgetDistributor):
        distr: dict[AutomatedStrategy : float] = distributor(self.bots, self.budget)
        for bot, budg in distr:
            bot.set_budget()