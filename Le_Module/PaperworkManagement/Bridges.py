import numpy as np
from Utilities.FinUtils import *
from tinkoff.invest import Client, OrderDirection, OrderType, CandleInstrument, InfoInstrument, SubscriptionInterval
import tinkoff.invest.services as ti_serv
from tinkoff.invest import InvestError
import uuid
from StrategiesManagement.sm import *
from PaperworkManagement.papman import *

class PaperBridge(Bridge):
    '''A bridge that executes strategys decisions with tinkoff invest sdk'''
    def __init__(self, token: str, acc_id: str, candle_instruments=list[CandleInstrument]):
        figis = []
        self.intervals = []
        for each in candle_instruments:
            figis.append(each.figi)
            self.intervals.append(each.interval)
        if len(set(figis)) != 1:
            print('WARNING: only the first figi will be interacted with') 
        self.token = token
        self.acc_id = acc_id
        self.bridge_streams = []
        with Client(token = self.token) as client:
            for each in candle_instruments:
                bridge_stream: ti_serv.MarketDataStreamManager = client.create_market_data_stream()
                bridge_stream.candles.waiting_close().subscribe(candle_instruments)
                self.bridge_streams.append(bridge_stream)
    
    def post_order(self, decision: Decision, figi: str):  
        try:
            with Client(token=self.token) as client:
                if decision.direction == True:
                    direction = OrderDirection.ORDER_DIRECTION_BUY
                elif decision.direction == False:
                    direction = OrderDirection.ORDER_DIRECTION_SELL
                    
                ordtypes = {0: OrderType.ORDER_TYPE_MARKET, 1: OrderType.ORDER_TYPE_BESTPRICE, 2: OrderType.ORDER_TYPE_LIMIT, 3: OrderType.ORDER_TYPE_UNSPECIFIED}
                ordertype = ordtypes[decision.type]
                
                orderprice = self.current_price() if decision.price == -1 else decision.price
                ord_id = str(uuid.uuid4())
                    
                order = client.orders.post_order(
                    figi=figi,
                    quantity=decision.amount,
                    price=Quotation(units=m.floor(orderprice), nano=(10**9)*orderprice%1),
                    direction=direction,
                    order_type=ordertype,
                    order_id=ord_id,
                    account_id=self.acc_id
                )
                
                return {'amount': decision.amount, 
                        'price': orderprice, 
                        'direction': decision.direction, 
                        'order id': ord_id, 
                        'order type': ordertype,
                        'figi': figi}
        except InvestError:
            print(f'for some reason could not post order {ord_id}, full decision: {decision}')
        
    def cancel_orders(self, order_ids):
        with Client(token=self.token) as client:
            for order in order_ids:
                try:
                    client.orders.cancel_order(account_id=self.acc_id, order_id=order)
                    print(f'order {order} cancelled successfully')
                except InvestError:
                    print(f'for some reason could not cancel order {order}')
        
    def current_price(self):
        with Client(token=self.token) as client:
            return client.market_data.get_last_prices(figi=self.figis)