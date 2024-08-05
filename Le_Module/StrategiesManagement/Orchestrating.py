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
    
    def __call__(self, bots: list[AutomatedStrategy], budget: float) -> dict[AutomatedStrategy : float]:
        pass  #should be defined by the user
            
class Orchestrator:
    def __init__(self, bots: list[AutomatedStrategy], total_budget: float, budget_distributor: BudgetDistributor):
        self.bots = bots
        for bot in bots: bot.orchestrator = self
        self.budget = total_budget
        self.distributor = budget_distributor
        self.already_asked = []
        
    def _distrib_budget(self, distributor: BudgetDistributor):
        self.budget = sum([bot.get_params.Budget for bot in self.bots])
        raw_distr = distributor(self.bots, self.budget)
        distr: dict[str : float] = {bot.strategy_id : raw_distr[bot]*self.budget for bot in raw_distr}
        
        assert sum(list(distr.values())) == self.budget
        return distr
    
    def __call__(self, id: str) -> dict | None:
        self.already_asked += [id]
        if len(set(self.already_asked)) == len(self.bots):
            self.already_asked = []
            print('\nbudget redistributed\n')
            return self._distrib_budget(self.distributor)
        else: return 