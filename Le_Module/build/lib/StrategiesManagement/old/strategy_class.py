#here is all the logic behind strategic decision making
import alive_progress
import time

class strategy_1:
    def __init__ (self, squad, n=1): #how_much_to_buy_or_sell_all
        self.squad = squad
        self.n = n

    def hmbsa(self, bit_of_data, price):
        ps = [m.make_prediction(bit_of_data) for m in self.squad['predictors']] 
        shares = 0
        N = 1
        if ps[0][1] > price*1.0005:
            for p in range(1, len(ps)):
                if ps[p][1] > ps[p-1][1]:
                    N = p+1
        
        if N > 1: #buy in advance
            shares += self.n*N
        if ps[0][0] > price*1.0005: # buy for the next day
            shares += 2*self.n
        elif ps[0][1] > price*1.0005 and ps[0][0] <= price*1.0005: #buy if the next step everything is not going to be so bright
            shares += 1*self.n
        if ps[0][1] < price*1.0005: #sell all if the next day the price is expected to fall the next step
            return 0 #shares to sell

        return shares #shares to buy
    
    
class risk_manager_atr:
    def __init__(self, atr_n):
        self.atr_n = atr_n

    def prevent_losses(self, buy_price, current_price, atr):
        if buy_price == 0 or current_price - buy_price < self.atr_n*atr:
            return False
        else:
            return True #if true - sell all immediately
        
class no_risk_management:
    def __init__(self):
        return None
    def prevent_losses():
        return None


class testing_agent:
    def __init__(self, squad, strategy, risk_manager, data, budget=0, reality=False):
        self.data = data #tuple of training data bits as after predictor_class.get_trainingdata()
        self.budget = budget #just a number, int or float
        self.shares = 0

        self.strat = strategy #strategy - obj of class <strategy_n>
        self.risk_manager = risk_manager #risk_manager - obj of class <risk_manager>
        self.bought_shares_log = []
        self.money_spent_before_sell = []

        self.squad = squad #squad - dict of models of class <predictor> (predictors, trend_viewers)
        assert len(squad['predictors']) != 0
        
        self.reality = reality
        self.pnls = []
        
    def test_strat(self):
        buy_price = None
        with alive_progress.alive_bar(len(self.data[0])-1, force_tty=True, spinner='stars') as bar:
            for i in range(len(self.data[1])):
                time.sleep(0.0005)
                data_bit = self.data[0][i:i+1]
                price = self.data[1][i-1].tolist()
                # right now is the i moment, in the labels_s` realm it`s i-1st moment
                shares_now = self.strat.hmbsa(data_bit, price)
                atr = data_bit[-1][-1].tolist()[0]
                if len(self.bought_shares_log) != 0:
                    buy_price = sum(self.money_spent_before_sell) / sum(self.bought_shares_log)
                else:
                    buy_price = 0

                not_allow_tade = False
                if len(self.risk_manager.prevent_losses.__code__.co_varnames) == 0:
                    not_allow_tade = False
                elif 'atr' in self.risk_manager.prevent_losses.__code__.co_varnames:
                    not_allow_tade = self.risk_manager.prevent_losses(buy_price, price, atr)

                if self.reality == True:
                    if not_allow_tade:
                        self.budget += self.shares*price*0.9995 # sell all
                        self.shares = 0
                        self.bought_shares_log = []
                        self.money_spent_before_sell = []

                    else:
                        if self.budget >= shares_now*price*1.0005:
                            if shares_now == 0:
                                self.budget += self.shares*price*0.9995 #sell all
                                self.shares = 0
                                self.bought_shares_log = []
                                self.money_spent_before_sell = []
                            else:
                                self.shares += shares_now
                                self.budget -= shares_now*price*1.0005
                                self.bought_shares_log.append(self.shares)
                                self.money_spent_before_sell.append(shares_now*price*1.0005)
                                
                        else:
                            self.budget = self.pnls[-1]   
                              
                else:
                    if not_allow_tade:
                        self.budget += self.shares*price*0.9995 # sell all
                        self.shares = 0
                        self.bought_shares_log = []
                        self.money_spent_before_sell = []
                    else:
                        if shares_now == 0:
                            self.budget += self.shares*price*0.9995 # sell all
                            self.shares = 0
                            self.bought_shares_log = []
                            self.money_spent_before_sell = []
                            
                        else:
                            self.shares += shares_now
                            self.budget -= shares_now*price*1.0005
                            self.bought_shares_log.append(self.shares)
                            self.money_spent_before_sell.append(shares_now*price*1.0005)
                self.pnls.append(self.budget)
                bar()