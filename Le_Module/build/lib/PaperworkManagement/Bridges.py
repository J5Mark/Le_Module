import numpy as np
#from StrategiesManagement import Decision
from Utilities.FinUtils import Candles
from tinkoff.invest import Client, OrderDirection, OrderType, CandleInstrument, InfoInstrument, SubscriptionInterval
import tinkoff.invest.services as ti_serv
import uuid
#import StrategiesManagement as sm

class PaperBridge:
    '''A bridge that executes strategys decisions with tinkoff invest sdk'''
    def __init__(self, token: str, acc_id: str, candle_instruments=[CandleInstrument]):
        self.figis = []
        self.intervals = []
        for each in candle_instruments:
            self.figis.append(each.figi)
            self.intervals.append(each.instrument)
        self.token = token
        self.acc_id = acc_id
        with Client(TOKEN = self.token) as client:
            bridge_stream: ti_serv.MarketDataStreamManager = client.create_market_data_stream()
            bridge_stream.candles.waiting_close().subscribe(candle_instruments)
        return bridge_stream
    
    def post_order(self, decision):  ###to avoid circular imports its worth to make up separate files foe different stuff of close or supportive to each other functionality
        with Client(token=self.token) as client:
            if decision.direction == True:
                direction = OrderDirection.ORDER_DIRECTION_BUY
            elif decision.direction == False:
                direction = OrderDirection.ORDER_DIRECTION_SELL
                
            ordtypes = {0: OrderType.ORDER_TYPE_MARKET, 1: OrderType.ORDER_TYPE_BESTPRICE, 2: OrderType.ORDER_TYPE_LIMIT, 3: OrderType.ORDER_TYPE_UNSPECIFIED}
            ordertype = ordtypes[decision.type]
            
            orderprice = self.current_price() if decision.price == -1 else order.price
                
            order = client.orders.post_order(
                figi=self.security,
                quantity=decision.amount,
                price=orderprice,
                direction=direction,
                order_type=ordertype,
                order_id=str(uuid.uuid4()),
                account_id=self.acc_id
            )
            
            return {'amount': order.quantity, 'price': order.price, 'direction': order.direction, 'order id': order.order_id}
        
    def cancel_order(self, order_id):
        pass
        
    def get_lot_size(self) -> int:
        pass
    def current_price(self):
        with Client(token=self.token) as client:
            return client.market_data.get_last_prices(figi=self.figis)