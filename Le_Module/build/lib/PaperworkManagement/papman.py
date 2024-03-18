from abc import ABC, abstractmethod
from tinkoff.invest import *
from Utilities.DataClasses import *


class Bridge(ABC):
    token: str
    acc_id: str
    candle_instruments: list
    
    @abstractmethod
    def post_order(self) -> Order:
        pass
    
    @abstractmethod
    def cancel_orders(self) -> None:
        pass
    
    @abstractmethod
    def current_price(self) -> Quotation:
        pass
    
    @abstractmethod
    def check_orders(self) -> list[OrderState]:
        pass