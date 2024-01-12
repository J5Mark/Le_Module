from abc import ABC, abstractmethod

class Bridge(ABC):
    token: str
    acc_id: str
    candle_instruments: list
    
    @abstractmethod
    def post_order(self):
        pass
    
    @abstractmethod
    def cancel_orders(self):
        pass
    
    @abstractmethod
    def current_price(self):
        pass