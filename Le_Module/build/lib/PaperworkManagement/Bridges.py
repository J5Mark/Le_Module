import numpy as np
from Utilities.FinUtils import *
from Utilities.DataClasses import *
from tinkoff.invest import Client, OrderDirection, OrderType, CandleInstrument, InfoInstrument, SubscriptionInterval, StopOrderType
import tinkoff.invest.services as ti_serv
from tinkoff.invest import InvestError, OrderState
import uuid
from StrategiesManagement.sm import *
from PaperworkManagement.papman import *

ordtypes = {0: OrderType.ORDER_TYPE_MARKET, 1: StopOrderType.STOP_ORDER_TYPE_STOP_LOSS, 2: StopOrderType.STOP_ORDER_TYPE_TAKE_PROFIT}

class CandlesBridge(Bridge):
    '''A bridge that executes strategys decisions with tinkoff invest sdk'''
    def __init__(self, token: str, acc_id: str, candle_instruments: list[CandleInstrument], waiting_close: bool=False):
        figis = []
        self.intervals = []
        for each in candle_instruments:
            figis.append(each.figi)
            self.intervals.append(each.interval)
        if len(set(figis)) != 1:
            print('WARNING: only the first figi will be interacted with') 
        self.token = token
        self.acc_id = acc_id
        self.bridge_streams = []   ####idk these fuckers dont work for some reason exiting with "ValueError: Cannot invoke RPC: Channel closed!"
        with Client(token = self.token) as client:
            for each in candle_instruments:
                bridge_stream = client.create_market_data_stream()
                bridge_stream.candles.waiting_close(enabled=waiting_close).subscribe([each])
                self.bridge_streams.append(bridge_stream)
    
    def post_order(self, decision: Decision, figi: str) -> Order:  
        try:
            with Client(token=self.token) as client:
                if decision.direction == True:
                    direction = OrderDirection.ORDER_DIRECTION_BUY
                elif decision.direction == False:
                    direction = OrderDirection.ORDER_DIRECTION_SELL
                    
                ordertype = ordtypes[decision.type]
                
                orderprice = self.current_price(figi) if decision.price == -1 else Quotation(units=m.floor(decision.price), nano=(10**9)*decision.price%1)
                ord_id = str(uuid.uuid4())
                    
                order = client.orders.post_order(
                    figi=figi,
                    quantity=decision.amount,
                    price=orderprice,
                    direction=direction,
                    order_type=ordertype,
                    order_id=ord_id,
                    account_id=self.acc_id
                )
                
                return Order(direction = decision.direction,
                             amount = decision.amount,
                             price=orderprice.units+orderprice.nano*10**(-9),
                             order_id=ord_id,
                             order_type=decision.type,
                             figi=figi)
        except InvestError:
            print(f'for some reason could not post order {ord_id}, full decision: {decision}')
            
    def check_orders(self, orders: list[Order]) -> list[OrderState]:
        with Client(token=self.token) as client:
            responses: list[OrderState] = []
            for ord in orders:
                responses.append(client.orders.get_order_state(ord.order_id))
                
        return responses
        
    def cancel_orders(self, order_ids):
        with Client(token=self.token) as client:
            for order in order_ids:
                try:
                    client.orders.cancel_order(account_id=self.acc_id, order_id=order)
                    print(f'order {order} cancelled successfully')
                except InvestError:
                    print(f'for some reason could not cancel order {order}')
        
    def current_price(self, figi) -> Quotation:
        try:
            with Client(token=self.token) as client:
                return client.market_data.get_last_prices(figi=[figi]).last_prices[-1].price
        except ValueError:
            print(f'seems like {figi} is not in the list of bridge`s figis.\nthese are in the list: {self.candle_instruments}')