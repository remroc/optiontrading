import sys
sys.path.append("API_tools")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import math_calc as mc
import time
import talib as ta
from datetime import datetime, date, timedelta
import API_basic as bapi
from ibapi.contract import Contract
import sqlite3

connection=sqlite3.connect("./Data/option_database/option.db")

#The siez of an option lot
lot_size=100

# Number of days needed to analysis underlying trend before getting a position
analisis_time=30
rolling_window=30

# Put True if you want to print graphs
graphs = False

symbol_list= ['ABNB','AVGO','MDLZ','AAPL','MSFT','AMZN','META','PH','AJG','CI','UPS','TDG','CDNS','CVS','FTNT','AON','NKE','NVDA',
            'RSG','ORLY','MMM','DELL','APO','ZTS','ECL','SNPS','GOOGL','TSLA','PYPL','MSI','RCL','GD']

# Dataframe used to store the results of the backtesting
results=pd.DataFrame()

# Define stop loss and take profit percentage
stop_loss_percentage = 1
tp_percentage = 1

def search_mult(symbol):
    """For a symbol in a str format, the function return the difference between
    the option strike prices for this symbol"""

    # The list of possiblities for the difference between option strike prices
    mult_option_possibilities = [10,5,2.5,1]

    # Getting the list of the strikes for the symbol
    data= pd.read_sql_query(f"SELECT strike FROM {symbol} GROUP BY strike",connection)

    # Useful constances
    flag = True
    i = 0
    while flag and i < 4:
        # We take a possibility in the liste
        mult_opt =mult_option_possibilities[i]
        flag = False

        # We check for all strikes if it is divisible by mult_opt
        for x in data['Strike']:
            if int(x / mult_opt) != (x/mult_opt):

                # If one strike is not divisible we continue by swtiching the flag
                flag = True

        # We pass to the next possibility
        i +=1
    return mult_opt

def get_premium(strike, expiration,type,date):
    """This fonction return the premium of the option searched
    - strike (int): strike price
    - expiration (str): day of expiration of the option on format YYYYMMDD
    - type (str): 'call' or 'put'
    - date (str): day searched for premium value on format YYYYMMDD"""
    
    # Getting data on a DataFrame
    data= pd.read_sql_query(f"SELECT * FROM {symbol} WHERE strike={strike} AND expiration={expiration} AND type='{type}'",connection)
    data['Date']=data['Date'].astype(str)
    data.set_index('Date',drop=True,inplace=True)
    data = data.drop('index',axis=1)

    # Checking if the DataFrame is not empty
    if len(data.index)==0:
        return None
    
    # If we have not the price for day searched, we take the last value which we have
    while not(date in data.index):
        date_num=(datetime.strptime(date,"%Y%m%d")-timedelta(1))
        date = datetime.strftime(date_num,"%Y%m%d")

        # If we don't have price for the option defined we return None and the program will make what needed
        if date == "20230101":
            return None
        

    return data.loc[date,'Close'].item()
    
def add_to_mtm(mark_to_market, option_strategy, trading_date):
    """Add the option_strategy to the mark_to_market Dataframe which
    store all the summary of trades did during backtestting.
    - mark_to_market (DataFrame): storing trades
    - option_strategy (DataFrame): dataframe representing the strategie
    - trading_date (str): day of the trade of YYYYMMDD format"""
    option_strategy['Date'] = trading_date
    mark_to_market = pd.concat([mark_to_market, option_strategy])
    return mark_to_market    

def get_IV_percentile(futures_data, rolling_window):
    """Fontion returing a Series of IV percentile.
    - futures_data (DataFrame): with a colum of IV (in percentage)
    - rolling_window (int): number of days taken in account to calculate IV percentile"""
    
    # Import required libraries
    from scipy import stats

    # Calculate IVP using the fonction provided by stat library and the data of futures_data
    futures_data['IV_percentile'] = futures_data['IV'].rolling(
        rolling_window).apply(lambda x: stats.percentileofscore(x, x[-1]))

    # Return IVP
    return futures_data['IV_percentile']

def setup_spread(futures_price,expiration,date, direction='long'):
    """Fonction returning a DataFrame representing a bull/bear spread strategie
    with type, strike,premium and position for alla four option of the spread.
    If direction is 'long' a bull spraed is set and a bear spread in the other case"""
    spread = pd.DataFrame()
    # Setting up the options type
    if direction =='long':
        spread['Option Type'] = ['C']
    else:
        spread['Option Type']=['P']
    
    # Setting the values for the ATM option and position
    atm_strike_price = mult_option * (round(futures_price / mult_option))
    spread['Strike Price'] = atm_strike_price
    spread['position'] = 1

    # Collecting the premium for ATM option
    for i in spread.index:
        spread.loc[i,'premium'] = \
            get_premium(spread.loc[i,'Strike Price'].item(),expiration,spread.loc[i,'Option Type'],date)

    # If we don't have the premium needed we fill a fake spread and the necessary will be did after
    if (spread.premium.isna().sum() > 0):
        spread.loc['1'] = ['C', atm_strike_price, -1, np.nan]
    else:

        # If all is good we set OTM option parameter
        deviation = max(round(spread.premium.sum()/mult_option),1) * mult_option
        if direction =='long':
            spread.loc['1'] = ['C', atm_strike_price+deviation, -1, np.nan]
        else:
            spread.loc['1'] = ['P', atm_strike_price-deviation, -1, np.nan]

    # Finding the premium for the OTM option
    spread.loc['1','premium'] = \
        get_premium(spread.loc['1','Strike Price'].item(),expiration,spread.loc['1','Option Type'],date)
    
    # Calculating total premium
    spread['premium'] = spread['premium'] * lot_size
    
    # Calculating the cost of the position taking margin account
    spread["cost"]= np.where(spread['position']==1,spread['premium'],spread['Strike Price']*lot_size*0.048 - spread['premium'])

    return spread

def dernier_vendredi_du_mois(date):
    """For a date at format of datetime library returning the date of last 
    Friday of the month of the date entered at same format."""

    # Finding the first day of the next month
    if date.month == 12:
        # If we are in december we go to junuary of the next year
        mois_suivant = datetime(date.year + 1, 1, 1)
    else:
        mois_suivant = datetime(date.year, date.month + 1, 1)
    
    # Getting the last day of current month
    dernier_jour_du_mois = mois_suivant - timedelta(days=1)
    
    # Going back to the last friday (weekday() == 4)
    while dernier_jour_du_mois.weekday() != 4:
        dernier_jour_du_mois -= timedelta(days=1)
    
    return dernier_jour_du_mois

for i in range(len(symbol_list)):

    # Definition of the variable used during backtesting
    symbol = symbol_list[i].lower()
    mult_option = search_mult(symbol)

    # List storing the values of losses and gains and cost of trades
    loses=[]
    gains=[]
    costs=[]

    trade_num = 0       # count the number of trades made
    sucess_trade=0      # count the number of trades successful
    position=0
    cum_pnl=0 
    exit_flag=False     # have value True when we have to exit from our position
    current_position=0  # take value 1 when we enter in position
    put_winner = 0        # Count the number of exits due to stop loss
    nb_put = 0

    # DataFrames which wille be use to store information during backtesting
    mark_to_market=pd.DataFrame() 
    round_trips_details = pd.DataFrame()    # to store the trades at the end of each trade
    trades = pd.DataFrame()                 # to store trades information during the trade

    # Connection to API
    app=bapi.Ib_user('127.0.0.1',7497,9)
    time.sleep(2)
    app.reset_data()
    time.sleep(0.1)

    # Definition of contract underlying
    underlying =Contract()
    underlying.symbol = symbol
    underlying.currency = "USD"
    underlying.exchange="SMART"
    underlying.secType="STK"

    # Collecting price historical data for the underlying
    app.reqHistoricalData(app.nextId(),underlying,"20250616 08:00:00 US/Eastern","50 W","1 day","MIDPOINT",1,1,False,[])
    time.sleep(2)
    stock_data = app.data
    app.reset_data()
    time.sleep(0.1)

    #Disconnection from API
    app.disconnect()
    

    # Calculate ADX
    stock_data['ADX'] = ta.ADX(stock_data.High, stock_data.Low, stock_data.Close, timeperiod=10)

    # Calculate the mean price for the last 5 days
    stock_data['sma_5'] = stock_data.Close.rolling(5).mean()

    # Calculate the mean price for the last 10 days
    stock_data['sma_30'] = stock_data.Close.rolling(12).mean()

    # Calculating for every day of backtesting the day of expiration and the number of trades before expiration
    for x in stock_data.index:
        # Find the date correpsonding to the friday from the same week that the day in index
        dday = datetime.strptime(x,"%Y%m%d")
        jour_semaine = dday.weekday()
        jours_vers_vendredi = 4 - jour_semaine
        vendredi = dday + timedelta(days=(jours_vers_vendredi))

        lastvendredi=dernier_vendredi_du_mois(datetime.strptime(x,"%Y%m%d"))

        # You can change the following line to either backtest a weekly strategie or a monthly strategie
        stock_data.loc[x,'Expiry']= vendredi
        stock_data.loc[x,'days_to_expiry']= (stock_data.loc[x,'Expiry']- datetime.strptime(x,"%Y%m%d")).days

    # Checking the position of our prices compared to the 5 last days mean
    stock_data['under_m'] = False
    stock_data['over_m'] = False
    for i in range(22,stock_data.shape[0]):
        sup = True
        inf= True
        for k in range (5):
            if stock_data.loc[stock_data.index[i-k],'Close'] > stock_data.loc[stock_data.index[i-k],'sma_5']:
                inf = False
            else:
                sup = False
        stock_data.loc[stock_data.index[i],'under_m'] = inf
        stock_data.loc[stock_data.index[i],'over_m'] = sup

    """for i in range(22,stock_data.shape[0]):
        sup = 0
        inf= 0
        for k in range (5):
            if stock_data.loc[stock_data.index[i-k],'Low'] < stock_data.loc[stock_data.index[i-k-1],'Low']:
                inf += 1
            else:
                sup += 1
        stock_data.loc[stock_data.index[i],'under_m'] = inf >= 5
        stock_data.loc[stock_data.index[i],'over_m'] = sup >= 5"""

    # Condition sign of bullish market
    condition_1 = (stock_data['sma_5'] > stock_data['sma_30'])

    # ADX entry condition
    condition_2 = (stock_data['ADX'] >= 10 ) & (stock_data['ADX'] <= 40)

    # Condition sign of bearish market
    condition_3 = (stock_data['sma_5'] < stock_data['sma_30'])

    # Generate signal as 1 if ADX condition is true
    stock_data['signal'] = np.where( condition_2 & condition_1 , 1, np.nan)
    stock_data['signal'] = np.where( condition_2 & condition_3, -1, stock_data['signal'])

    # Generate signal as 0 on expiry dates (thursday and friday)
    stock_data['signal'] = np.where(stock_data['days_to_expiry'] <=1, 0, stock_data['signal'])

    # Finding the first date for testing the strategy after the analysis days
    start_date = datetime.strptime(stock_data.index[0],"%Y%m%d") + timedelta(days=30)

    # Index for the days we will try enter in position
    list_index = list(stock_data.loc[datetime.strftime(start_date,"%Y%m%d"):].index)

    # Represent the place of index in list_index
    a=-1

    for i in list_index[5:]:

        # We store the current position and the current PnL accumulated
        stock_data.loc[i,'Position'] = current_position
        stock_data.loc[i,'Total_pnl'] = cum_pnl

        # We enter in position if we have a signal indicating it
        """(stock_data.loc[i, 'signal'] != 0)"""
        if (current_position == 0) and ((stock_data.loc[i, 'signal'] != 0)):
            
            # Setup spread
            if stock_data.loc[i,'over_m'] and stock_data.loc[i,'signal'] == 1: 
                spread = setup_spread(stock_data.loc[i,'Close'].item(),datetime.strftime(stock_data.loc[i,'Expiry'],"%Y%m%d"),i,direction = "long")
            elif stock_data.loc[i,'under_m'] and stock_data.loc[i,'signal'] == -1:
                spread = setup_spread(stock_data.loc[i,'Close'].item(),datetime.strftime(stock_data.loc[i,'Expiry'],"%Y%m%d"),i,direction = "short")
            else:
                # If conditions are not met we pass to the next day
                continue
            
            # Check that the last price of all the legs of the butterfly is greater than 0
            # If it is not the case, it's that the data of premiums have not be found
            if (spread.premium.isna().sum() > 0) or ((spread.premium == 0).sum() > 0):
                print(spread)
                print(f"\x1b[31mStrike price is not liquid so we will ignore this trading opportunity {i}\x1b[0m")
                continue
            
            # Fill the trades dataframe that will be used to make calculation on the exit day
            trades = spread.copy()
            trades['entry_date'] = i
            trades.rename(columns={'premium':'entry_price'}, inplace=True)           
            
            # Calculate net premium and the cost of the trade
            net_premium = round((spread.position * spread.premium).sum(),1)
            cost = round((spread.cost).sum(),1)
            costs.append(cost)

            # Define the stop loss and take profit
            sl = net_premium*(1 - stop_loss_percentage)
            tp = net_premium*(1 + tp_percentage)

            # Update current position to 1
            current_position = 1
            
            # Update mark_to_market dataframe
            mark_to_market = add_to_mtm(mark_to_market, spread, i)
            
            # Increase number of trades by 1
            trade_num += 1
            
            # Print trade details
            print("-"*30)
            print(spread)
            print(f"Trade No: {trade_num} | Entry Date: {i} | Premium: {net_premium}",
                f"|Underlying: {stock_data.loc[i,'Close'].item()} |Cost: {cost}| ADX:{stock_data.loc[i,'ADX'].item()}")           
                
        elif current_position == 1:
            
            # The premium is updated in the case where we have to exit to take for Stop Loss or Take Profit
            # Update net premium
            for k in spread.index:
                spread.loc[k,'premium'] = \
                    get_premium(spread.loc[k,'Strike Price'].item(),datetime.strftime(stock_data.loc[i,'Expiry'],"%Y%m%d")
                                ,spread.loc[k,'Option Type'],i)
            spread['premium'] = spread['premium']*100
                
            # We calculte net premium
            net_premium = (spread.position * spread.premium).sum()

            # We print net_premium and underlying price to see the evloution of price
            print(f"date: {i} ", net_premium," ",stock_data.loc[i,'Close'].item())
            
            # Update mark_to_market dataframe
            mark_to_market = add_to_mtm(mark_to_market, spread, i)
        
            # Exit at expiry

            # On expiry date, stock_data.loc[i, 'signal'] is set to be 0 (it has been set before)
            if stock_data.loc[i, 'signal'] == 0:
                exit_type = 'Expiry'
                exit_flag = True 
                
            if exit_flag:
                
                # Check that the data is present for all strike prices on the exit date, if it is not 
                # there is a lack of data and we don't take this trade in account
                if spread.premium.isna().sum() > 0:
                    print(f"Data missing for the required strike prices on {i}, Not adding to trade logs.")
                    current_position = 0
                    continue
                
                # Update the trades dataframe
                trades['exit_date'] = i
                trades['exit_type'] = exit_type
                trades['exit_price'] = spread.premium
                
                # Add the trade logs to round trip details
                round_trips_details = pd.concat([round_trips_details,trades])
                
                # Calculate net premium at exit
                net_premium = round((spread.position * spread.premium).sum(),1)   
                
                # Calculate net premium on entry
                entry_net_premium = (trades.position * trades.entry_price).sum()       
                
                # Calculate pnl for the trade
                trade_pnl = round(net_premium - entry_net_premium,1)
                
                # Calculate cumulative pnl
                cum_pnl += trade_pnl
                cum_pnl = round(cum_pnl,2)
                
                # Print trade details
                print(f"""Trade No: {trade_num} | Exit Type: {exit_type} | Date: {i} | Premium: {net_premium} | PnL: {trade_pnl}
                    | Cum PnL: {cum_pnl}|Underlying: {stock_data.loc[i,'Close'].item()}""")                              

                # Update current position to 0
                current_position = 0    
                
                # Set exit flag to false
                exit_flag = False

                # We save the results of the backtesting to make some comparaison
                if trade_pnl > 0:
                    sucess_trade +=1
                    gains.append(trade_pnl)
                    if spread.loc['1','Option Type'] == 'P':
                        put_winner +=1
                else:
                    loses.append(trade_pnl)

                if spread.loc['1','Option Type'] == 'P':
                        nb_put +=1


    # We calculate the analysis variables
    n_res = pd.DataFrame({"Symbol":[symbol],"PnL":[cum_pnl],"Number of trades":[trade_num],"Winners":[sucess_trade],
                        "Losers":[trade_num - sucess_trade]})
    n_res["Win_percentage"] = n_res["Winners"] / n_res["Number of trades"]
    n_res["Average gain"] = np.array(gains).mean()
    n_res["Average lose"] = np.array(loses).mean()
    n_res["Profit factor"] = abs((n_res['Win_percentage'] * n_res['Average gain']) \
        /((1 - n_res['Win_percentage'])* n_res['Average lose']))
    n_res["Average cost"]=np.array(costs).mean()
    n_res["Return"] = (n_res["PnL"] / n_res["Average cost"])*100
    n_res['NB_call'] = nb_put
    n_res['Call winner'] = put_winner
    n_res.set_index("Symbol",drop=True,inplace=True)
    
    # We add the variable calculated at the DataFrames which summarise the backtest results
    results = pd.concat([results,n_res])

    # If wanted we show some graphs
    if graphs:
        plt.plot(stock_data['Close'], label = 'close price')
        plt.plot(stock_data['Position'] * stock_data['Close'].max())
        plt.plot(stock_data['sma_5'],label='sma_5')
        plt.plot(stock_data['sma_30'], color = 'blue', label='sma_30')
        plt.legend()
        plt.show()
        plt.plot(stock_data['Total_pnl'])
        plt.show()

# We print summay of our global results
print(results)
print(f"Total PnL: {results['PnL'].sum()}")
print(f"Win % :{results['Winners'].sum()*100 / results['Number of trades'].sum()}")
print(f"Bear Win%: {results['Call winner'].sum()*100 / results['NB_call'].sum()}")
print(f"Profit factor: {results['Profit factor'].mean()}")
print(f"Average return: {results['Return'].mean()}")