# Version: 0.01
# API doc: https://binance-docs.github.io/apidocs/spot/en/#change-log
# todo: exceptions in while loop
# todo: rollback when db error: https://github.com/mkleehammer/pyodbc/wiki/Cursor
# todo: try and exceptions in loop when doing on DB
# todo: error logs insert while exception. Maybe into text file, because error can be on DB
# todo: conn close each time
# todo: synhronize time with time.nist.gov,
# todo: maybe not when there is simestamp < dell, but when new record is in json from binance
# todo: może warto wyprzedzić i brać sygnał z godziny xx:xx:59 wg. ostatniego rekordu
# todo: db timeout and connections close
# todo: v0.02: handle error when account has not enought cash to do buy order

# get libs
import time
from datetime import datetime
import requests  # https://docs.python-requests.org/en/master/
import pandas as pd
import pandas_ta as pta
import uuid  # https://docs.python.org/3/library/uuid.html
import json
from binance.client import Client
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager  # (lib: python-binance)
from binance.enums import *
import os

# settings
WORKER_ID = 1
WORKLOGS_DIR = "worklogs"
worker_start_time = str(datetime.utcnow())

# empty variables
signal_params = {}
signal_expiration_timestamp = 0
signal_list = []
external_open_orders_list = []  # imported from stock exchange only for open orders

# create temporary directory for downloaded files
def create_temp_dir(worklogs_dir_in):
    try:
        os.mkdir(worklogs_dir_in)
        print("Worklogs DIR created")
    except OSError as error:
        print(error)


# worklogs
def stock_play_worklog_to_json(wlog_string):
    # todo: create dir if not exist
    with open(WORKLOGS_DIR + '//' + 'worklog_'+str(time.time())+'.json', 'w') as f:
        json.dump(wlog_string, f)


# get game config from json file
def get_game_settings_from_json():
    with open("game_conf.json") as json_conf:
        game_conf = (json.load(json_conf))
    print("conf file opened")
    return game_conf


# tactic assignment - game config
def assign_game_params():
    game_conf = get_game_settings_from_json()
    tactic_id = game_conf["tactic_id"]
    tactic_name = game_conf["tactic_name"]
    buy_indicator_1_signal_name = game_conf["buy_indicator_1_signal_name"]
    buy_indicator_1_signal_value = game_conf["buy_indicator_1_signal_value"]
    signal_expiration_time = game_conf["signal_expiration_time"]
    signal_sell_expiration_time = game_conf["signal_sell_expiration_time"]
    signal_sell_yield_expected = game_conf["signal_sell_yield_expected"]
    single_game_stake = game_conf["single_game_stake"]
    market = game_conf["market"]
    tick_interval = game_conf["tick_interval"]
    data_granulation = game_conf["data_granulation"]
    stock_type = game_conf["stock_type"]
    stock_exchange = game_conf["stock_exchange"]

    print("Game config assignment done")
    return tactic_id, tactic_name, buy_indicator_1_signal_name,buy_indicator_1_signal_value, signal_expiration_time,\
           signal_sell_expiration_time, signal_sell_yield_expected, single_game_stake, market, tick_interval,\
           data_granulation, stock_type, stock_exchange

tactic_id, tactic_name, buy_indicator_1_signal_name,buy_indicator_1_signal_value, signal_expiration_time,\
           signal_sell_expiration_time, signal_sell_yield_expected, single_game_stake, market, tick_interval,\
           data_granulation, stock_type, stock_exchange = assign_game_params()


def get_binance_current_data():
    url = 'https://api.binance.com/api/v3/klines?symbol=' + market + '&interval=' + tick_interval
    try:
        data = requests.get(url).json()
    except:
        print("error occured in get_binance_current_data()")
        time.sleep(86400 / 24 / 60 / 60 * 0.05)
        data = requests.get(url).json()
        # todo: do error log
    return data


def get_binance_client_connection():
    # get binance credentials from stored file
    with open('stock_conf.json') as json_stock_conf:
        stock_config = (json.load(json_stock_conf))
    api_key = stock_config["api_key"]
    api_secret = stock_config["api_secret"]
    # binance client connection
    client = Client(api_key, api_secret)
    return client

client = get_binance_client_connection()



# main loop start
iterator = 0.0  # main loop iterator
signal_generation_status = False


# todo: get last signal from DB if exist. Use strategy name to recognise signal (while fuckup). Need stock time. Copy from start_old.py

# todo: cancel all buy orders immediately at start. According to this application. It will cancel also orders from other applications. Delete or Improve when DB

def delete_active_buy_orders():
    external_open_orders_list = client.get_open_orders(symbol=market)
    for j in external_open_orders_list:
        if j["status"] in ("NEW", "PARTIALLY_FILLED") and j["side"] == "BUY":
            cancel_order = client.cancel_order(
                symbol=market,
                orderId=j['orderId'])
            #print(corder)


if __name__ == "__main__":
    create_temp_dir(WORKLOGS_DIR)

    delete_active_buy_orders()

    x = 1
    while x == x:

        # todo: DONE get time from stock exchanege
        # todo: create function
        # server time
        stock_time = client.get_server_time()['serverTime'] / 1000
        # print(stock_time)

        # todo: DONE crash expired signals on python only
        if (signal_generation_status == True) and (signal_expiration_timestamp < stock_time):
            signal_generation_status = False
            signal_params = ''
            print("old signal crashed")

        # todo: DONE Signal params generation. Do only when there is no active signals
        if signal_generation_status == False:
            data = get_binance_current_data()

            df = pd.DataFrame(data, columns=["open_time", "open", "high", "low", "close", "volume", "close_time",
                                                      "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
                                                      "taker_buy_quote_asset_volume", "ignore"], dtype=float)

            # basics (not needed)
            df["open_time_dt"] = pd.to_datetime(df["open_time"], unit='ms')
            df["close_time_dt"] = pd.to_datetime(df["close_time"], unit='ms')
            df["change_val"] = df.close - df.open
            df["change_perc"] = df.close / df.open - 1
            df["amplitude_val"] = df.high - df.low
            df["amplitude_perc"] = df.high - df.low / df.open

            # indicators
            # RSI
            rsi_6 = pta.rsi(df['close'], length=6)
            df.append(rsi_6)
            # ROC
            indicator_1_buy = pta.roc(df['close'], length=24)  # todo: do smth. PTA and function write are not needed
            df.append(indicator_1_buy)



            # todo: DONE buy signal variables values ie. if indicator below x points
            # todo: signal_id need to be based on unix timestamp and tactic name and maybe worker - you can crash signals/orders in one instance of application
            if (indicator_1_buy.iloc[-2] < buy_indicator_1_signal_value):

                signal_id = str(uuid.uuid1())
                signal_timestamp = stock_time
                signal_datetime = str(datetime.fromtimestamp(signal_timestamp))
                signal_first_timestamp = float(df["open_time"].iloc[-1] / 1000)
                signal_first_datetime = str(datetime.fromtimestamp((df["open_time"].iloc[-1] / 1000)))
                signal_expiration_timestamp = float(df["open_time"].iloc[-1] / 1000) + signal_expiration_time
                signal_expiration_datetime = str(datetime.fromtimestamp(signal_expiration_timestamp))
                signal_sell_expiration_timestamp = float(df["open_time"].iloc[-1] / 1000) + signal_sell_expiration_time
                signal_sell_expiration_datetime = str(datetime.fromtimestamp(signal_sell_expiration_timestamp))
                signal_type = "BUY_SELL_SIGNAL"
                signal_buy_price = round(float(df["close"].iloc[-2]), 0) # int buy prices
                signal_sell_price = round(float(df["close"].iloc[-2] * signal_sell_yield_expected + df["close"].iloc[-2]), 0)  # int sell prices
                signal_tactic = tactic_name
                signal_market = market
                signal_tick_interval = tick_interval
                signal_buy_quantity = round(single_game_stake / round(float(df["close"].iloc[-2]), 0), 4)
                signal_status = "NEW"
                signal_buy_external_ordrid = ""
                signal_sell_external_ordrid = ""
                signal_indicators_values = {"indicator_1_buy": float(indicator_1_buy.iloc[-2])}


                # todo: DONE in case of time differences between servers. Don't take old signal
                if signal_expiration_timestamp > signal_timestamp:

                    signal_params = {"signal_id": signal_id,
                         "signal_timestamp": signal_timestamp,
                         "signal_datetime": signal_datetime,
                         "signal_first_timestamp": signal_first_timestamp,
                         "signal_first_datetime": signal_first_datetime,
                         "signal_expiration_timestamp": signal_expiration_timestamp,
                         "signal_expiration_datetime": signal_expiration_datetime,
                         "signal_sell_expiration_timestamp":signal_sell_expiration_timestamp,
                         "signal_sell_expiration_datetime": signal_sell_expiration_datetime,
                         "signal_type": "BUY_SIGNAL",
                         "signal_buy_price" : signal_buy_price,
                         "signal_sell_price" : signal_sell_price,
                         "signal_tactic" : tactic_name,
                         "signal_market" : market,
                         "signal_tick_interval" : signal_tick_interval,
                         "signal_buy_quantity" : signal_buy_quantity,
                         "signal_status": signal_status,
                         "signal_buy_external_ordrid": signal_buy_external_ordrid,
                         "signal_sell_external_ordrid": signal_sell_external_ordrid,
                         "signal_indicators_values": signal_indicators_values}
                    signal_generation_status = True

                    print("new buy signal generted:")
                    print(str(signal_params))

                    # todo: insert signal into valid signal lists or dict (can be False, or expired, but not sold)
                    signal_list.append(signal_params)

                    print("signal_list:")
                    print(signal_list)

                # todo: buy order create. After buy check fill. When fill Sell. When expired cancel order
                # todo: order params need to be in game json
                if signal_generation_status == True:
                    for i in signal_list:
                        if i["signal_status"] == "NEW":
                            buy_order = client.create_order(
                                symbol=market,
                                newClientOrderId=signal_id,
                                side=SIDE_BUY,
                                type=ORDER_TYPE_LIMIT,
                                timeInForce=TIME_IN_FORCE_GTC,
                                quantity=signal_buy_quantity,
                                price=str(signal_buy_price))  # delete 10000 on real game

                            # todo: add external order_id to signal_list
                            i["signal_buy_external_ordrid"] = buy_order["orderId"]
                            i["signal_status"] = "BUY ORDER CREATED"

                            print("buy order done:")
                            print(buy_order)

                            # todo: wait 1 sec after buy. Check leter for best time
                            # todo: change to try-except
                            time.sleep(86400 / 24 / 60 / 60 * 0.1)


        # todo: check filled orders. Create sell order
        for i in signal_list:
            if i["signal_buy_external_ordrid"] != '' and i["signal_status"] == "BUY ORDER CREATED":
                check_order = client.get_order(
                    symbol=market,
                    orderId=i["signal_buy_external_ordrid"])
                if check_order["status"] in ("FILLED", "PARTIALY_FILLED"):
                    i["signal_status"] = check_order["status"]

                    sell_order = client.order_limit_sell(
                        symbol=market,
                        quantity=signal_buy_quantity,
                        price=str(i["signal_sell_price"]))

                    #todo: add external order_id and status to signal_list
                    i["signal_sell_external_ordrid"] = sell_order["orderId"]
                    i["signal_status"] = "SELL ORDER CREATED"
                    print(sell_order)


        # todo: cancel expired buy orders based on signals
        external_open_orders_list = client.get_open_orders(symbol=market)
        for i in signal_list:
            if (i["signal_expiration_timestamp"]) < stock_time and i["signal_status"] in ("NEW", "PARTIALLY_FILLED"):
                for j in external_open_orders_list:
                    if (i["signal_id"] == j["clientOrderId"]) and j["side"] == "BUY":
                        # cancel order here:
                        cancel_order = client.cancel_order(
                            symbol=market,
                            orderId=j['orderId'])

                        i["signal_status"] = "CANCEL BUY / TIME EXPIRED"
                        print("cancel order:")
                        print(cancel_order)

        # todo: cancel expired buy orders based on time. In case of reset / shutdown
        external_open_orders_list = client.get_open_orders(symbol=market)
        for j in external_open_orders_list:
            if j["status"] in ("NEW", "PARTIALLY_FILLED") and j["side"] == "BUY" and j["time"] / 1000 + signal_expiration_time < stock_time:
                print(j["status"] + j["clientOrderId"])
                print(j)
                cancel_order = client.cancel_order(
                    symbol=market,
                    orderId=j['orderId'])
                print(cancel_order)


        # todo: market sell for expired sell orders (ie. after 30 min from buy sell does not exist.
        # cancel sell
        # do market sell order
        external_open_orders_list = client.get_open_orders(symbol=market)
        for i in signal_list:
            if (i["signal_sell_expiration_timestamp"]) < stock_time and i["signal_status"] == ("SELL ORDER CREATED"):
                for j in external_open_orders_list:
                    if (i["signal_sell_external_ordrid"] == j["orderId"]) and j["side"] == "SELL":
                        # cancel order here:
                        cancel_order = client.cancel_order(
                            symbol=market,
                            orderId=j['orderId'])

                        i["signal_status"] = "CANCEL SELL / TIME EXPIRED"
                        print("cancel order:")
                        print(cancel_order)
                        time.sleep(86400 / 24 / 60 / 60 * 0.5)  # improve this sleep, THIS IS SHIT, BUT NECESSARY. Sux when you have more than one order in same time

                        # todo: do market sell on expired sell order.
                        sell_order = client.order_market_sell(
                            symbol=market,
                            quantity=signal_buy_quantity)

                        # todo: add external order_id and status to signal_list
                        i["signal_sell_external_ordrid"] = sell_order["orderId"]
                        i["signal_status"] = "SELL ORDER MARKET CREATED BECAUSE SELL EXPIRED"
                        print(sell_order)



        # todo: DONE worklog - 1 of x operations
        iterator = iterator + 1.0

        if iterator % 100.0 == 0:
            print("heart beat: worker_id: " + str(WORKER_ID) + ", iteration: " + str(iterator) + ", tactic_id: " +
                  str(tactic_id) + ", indicator_1_buy_value: " + str(str(float(indicator_1_buy.iloc[-2]))) + ", UTC: " +
                  str(datetime.utcnow()))
            wlog_string = {"worker_id": WORKER_ID,
                           "worker_start_time": worker_start_time,
                           "iteration": iterator,
                           "tactic_id": tactic_id,
                           "indicator_1_buy_name": buy_indicator_1_signal_name,
                           "indicator_1_buy_value": buy_indicator_1_signal_value,
                           "indicator_real_value": float(indicator_1_buy.iloc[-2]),
                           "utc": str(datetime.utcnow())
                           }
            stock_play_worklog_to_json(wlog_string)

        # todo:


        # todo: sleep x seconds
        # time.sleep(86400/24/60/60*0.05)
