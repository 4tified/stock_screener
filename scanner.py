import tumonkey as ts
from pushbullet import Pushbullet
import time

class Scanning:
    def __init__(self, code, price_alarm):
        self.flag = 0
        self.code = code
        self.price_alarm = price_alarm
        self.pb = Pushbullet('7OcQOepNFS7w2vo0nuR32YKZ9Rv5sdpL')
        self.scanner()
        
    def scanner(self):
        df = ts.get_realtime_quotes(self.code)
        df[['code','name','price','bid','ask','volume','amount','time']]
        current_price =  df['price'][0]
        print current_price
        if current_price == self.price_alarm:
            print "Reached!"
            if self.flag == 0:
                push = self.pb.push_note("STOCK ALERT: " + str(self.code), "Price of: " + str(self.price_alarm) + " has been reached!")
                self.flag = 1
            elif self.flag == 1:
                quit()
        time.sleep(5)
        self.scanner()


code = raw_input('Stock Code: ')
price_alarm = raw_input('Price to alarm: ')
scan = Scanning(code, price_alarm)
