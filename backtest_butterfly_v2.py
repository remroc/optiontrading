import sys
sys.path.append("./../API_tools")
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

# Connection to the database storing 
connection=sqlite3.connect("./../Data/option_database/option.db")

#Size of an option lot
lot_size=100

# Number of days needed to analysis underlying trend before getting a position
analisis_time=30
rolling_window=30

# Put True if you want to print graphs
graphs = False

# List of the symbols on which we make the backtesting
symbol_list=['ABNB','AVGO','MDLZ','AAPL','MSFT','AMZN','META','PH','AJG','CI','UPS','TDG','CDNS','CVS','FTNT','AON','NKE','NVDA',
            'RSG','ORLY','MMM','DELL','APO','ZTS','ECL','SNPS','GOOGL','TSLA','PYPL','MSI','RCL','GD']

# Dataframe used to store the results of the backtesting
results=pd.DataFrame()

# Define the percentage stop loss and take profit percentage
stop_loss_percentage = 1
tp_percentage = 1

def search_mult(symbol):
    """For a symbol in a str format, the function return the difference between
    the option strike prices"""

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

        # We cannot get any price before this date 
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


def setup_butterfly(futures_price,expiration,date, direction='long'):
    """Fonction returning a DataFrame representing a short/long butterfly strategie
    with type, strike,premium and position for all four option of the butterfly"""
    butterfly = pd.DataFrame()

    # Setting the values for the two ATM option
    butterfly['Option Type'] = ['C', 'P']
    atm_strike_price = mult_option * (round(futures_price / mult_option))
    butterfly['Strike Price'] = atm_strike_price
    butterfly['position'] = 1

    # Collecting the premium for ATM option
    for i in butterfly.index:
        butterfly.loc[i,'premium'] = \
            get_premium(butterfly.loc[i,'Strike Price'].item(),expiration,butterfly.loc[i,'Option Type'],date)

    # If we have the premium needed we fill a fake butterfly and the necessary will be did after
    if (butterfly.premium.isna().sum() > 0):
        butterfly.loc['2'] = ['C', atm_strike_price, -1, np.nan]
        butterfly.loc['3'] = ['P', atm_strike_price, -1, np.nan]
    else:
        # If all is good we set OTM option parameter
        print("Passez par là")
        deviation = round(butterfly.premium.sum()/mult_option) * mult_option
        butterfly.loc['2'] = ['C', atm_strike_price+deviation, -1, np.nan]
        butterfly.loc['3'] = ['P', atm_strike_price-deviation, -1, np.nan]

    # Setting up the positions
    if direction == 'long':
        butterfly['position'] *= -1

    # Calculating the premium for the OTM options 
    for i in butterfly.index[2:]:
        butterfly.loc[i,'premium'] = \
            get_premium(butterfly.loc[i,'Strike Price'].item(),expiration,butterfly.loc[i,'Option Type'],date)
        
    # Multiplying the premium by the size of option lot
    butterfly['premium'] = butterfly['premium'] * lot_size
        
    # Calculating the cost of the butterfly taking account margin for short options
    butterfly["cost"]= np.where(butterfly['position']==1,butterfly['premium'],butterfly['Strike Price']*lot_size*0.048 - butterfly['premium'])

    return butterfly

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
    sl_count = 0        # Count the number of exits due to stop loss

    # DataFrames which wille be use to store information during backtesting
    mark_to_market=pd.DataFrame() 
    round_trips_details = pd.DataFrame()    # to store the trades at the end of each trade
    trades = pd.DataFrame()                 # to store trades information during the trade

    # Connection to API
    app=bapi.Ib_user('127.0.0.1',7497,9)
    time.sleep(0.1)
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
    time.sleep(1)
    stock_data = app.data
    app.reset_data()
    time.sleep(0.1)

    # Collecting the implied volatility
    app.reqHistoricalData(app.nextId(),underlying,"20250616 08:00:00 US/Eastern","50 W","1 day","OPTION_IMPLIED_VOLATILITY",1,1,False,[])
    time.sleep(1)

    # Putting the implied volatility on percentage format
    stock_data['IV']=app.data['Close']*100
    print(stock_data)

    #Disconnection from API
    app.disconnect()
    

    # Calculate ADX
    stock_data['ADX'] = ta.ADX(stock_data.High, stock_data.Low, stock_data.Close, timeperiod=14)

    # Calculate IVP
    stock_data['IVP'] = get_IV_percentile(stock_data, 30)

    # Calculating for every day of backtesting the day of expiration and the number of trades before expiration
    for x in stock_data.index:
        # Find the date correpsonding to the friday from the same week that the day in index
        dday = datetime.strptime(x,"%Y%m%d")
        jour_semaine = dday.weekday()
        jours_vers_vendredi = 4 - jour_semaine
        vendredi = dday + timedelta(days=jours_vers_vendredi)

        lastvendredi=dernier_vendredi_du_mois(datetime.strptime(x,"%Y%m%d"))

        # You can change the following line to either backtest a weekly strategie or a monthly strategie
        stock_data.loc[x,'Expiry']= vendredi
        stock_data.loc[x,'days_to_expiry']= (stock_data.loc[x,'Expiry']- datetime.strptime(x,"%Y%m%d")).days

    # IVP entry condition
    condition_1 = (stock_data['IVP'] >= 50) & (stock_data['IVP'] <= 95)

    # ADX entry condition
    condition_2 = (stock_data['ADX'] <= 30)


    condition_3 = (stock_data['IV'] >= 20)

    # Generate signal as 1 when both conditions are true
    stock_data['signal_adx_ivp'] = np.where( condition_1 & condition_2 & condition_3, 1, np.nan)

    # Generate signal as 0 on expiry dates
    stock_data['signal_adx_ivp'] = np.where(stock_data['days_to_expiry']<=1, 0, stock_data['signal_adx_ivp'])
    start_date = datetime.strptime(stock_data.index[0],"%Y%m%d") + timedelta(days=30)


    for i in stock_data.loc[datetime.strftime(start_date,"%Y%m%d"):].index:

        # We store the current position and the current PnL accumulated
        stock_data.loc[i,'Position'] = current_position
        stock_data.loc[i,'Total_pnl'] = cum_pnl

        # We enter in position if we have a signal indicating it
        if (current_position == 0) & (stock_data.loc[i, 'signal_adx_ivp'] == 1):
            
            # Setup butterfly
            butterfly = setup_butterfly(stock_data.loc[i,'Close'].item(),datetime.strftime(stock_data.loc[i,'Expiry'],"%Y%m%d"),i,direction = "short")
                
            # Check that the last price of all the legs of the butterfly is greater than 0
            # If it is not the case, it's that the data of premiums have not be found
            if (butterfly.premium.isna().sum() > 0) or ((butterfly.premium == 0).sum() > 0):
                print(butterfly)
                print(f"\x1b[31mStrike price is not liquid so we will ignore this trading opportunity {i}\x1b[0m")
                continue
            
            # Fill the trades dataframe that will be used to make calculation on the exit day
            trades = butterfly.copy()
            trades['entry_date'] = i
            trades.rename(columns={'premium':'entry_price'}, inplace=True)           
            
            # Calculate net premium and the cost of the trade
            net_premium = round((butterfly.position * butterfly.premium).sum(),1)
            cost = round((butterfly.cost).sum(),1)
            costs.append(cost)

            # Define the stop loss and take profit
            sl = net_premium*(1 - stop_loss_percentage)
            tp = net_premium*(1 + tp_percentage)

            # Update current position to 1
            current_position = 1
            
            # Update mark_to_market dataframe
            mark_to_market = add_to_mtm(mark_to_market, butterfly, i)
            
            # Increase number of trades by 1
            trade_num += 1
            
            # Print trade details
            print("-"*30)
            print(f"Trade No: {trade_num} | Entry | Date: {i} | Premium: {net_premium} \
                |Underlying: {stock_data.loc[i,'Close'].item()} |Cost: {cost}")           
                
        elif current_position == 1:
            
            # The premium is updated in the case where we have to exit to take for Stop Loss or Take Profit
            # Update net premium
            for k in butterfly.index:
                butterfly.loc[k,'premium'] = \
                    get_premium(butterfly.loc[k,'Strike Price'].item(),datetime.strftime(stock_data.loc[i,'Expiry'],"%Y%m%d")
                                ,butterfly.loc[k,'Option Type'],i)
            butterfly['premium'] = butterfly['premium']*100
                
            # We calculte net premium
            net_premium = (butterfly.position * butterfly.premium).sum()
            
            # Update mark_to_market dataframe
            mark_to_market = add_to_mtm(mark_to_market, butterfly, i)

            # On expiry date, stock_data.loc[i, 'signal_adx_ivp'] is set to be 0 (it has been set before)
            if stock_data.loc[i, 'signal_adx_ivp'] == 0:
                exit_type = 'Expiry'
                exit_flag = True 
                
            if exit_flag:
                
                # Check that the data is present for all strike prices on the exit date, if it is not 
                # there is a lack of data and we don't take this trade in account
                if butterfly.premium.isna().sum() > 0:
                    print(f"Data missing for the required strike prices on {i}, Not adding to trade logs.")
                    current_position = 0
                    continue
                
                # Update the trades dataframe
                trades['exit_date'] = i
                trades['exit_type'] = exit_type
                trades['exit_price'] = butterfly.premium
                
                # Add the trade logs to round trip details
                round_trips_details = pd.concat([round_trips_details,trades])
                
                # Calculate net premium at exit
                net_premium = round((butterfly.position * butterfly.premium).sum(),1)   
                
                # Calculate net premium on entry
                entry_net_premium = (trades.position * trades.entry_price).sum()       
                
                # Calculate pnl for the trade!!!!!!!!!!!
                trade_pnl = round(net_premium - entry_net_premium,1)

                # Add 1 to stop loss exit count if exit_type is stop loss
                if exit_type == "Expiry":
                    sl_count += 1
                
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
                else:
                    loses.append(trade_pnl)

    # We calculate the analysis variables
    n_res = pd.DataFrame({"Symbol":[symbol],"PnL":[cum_pnl],"Number of trades":[trade_num],"Winners":[sucess_trade],
                        "Losers":[trade_num - sucess_trade]})
    n_res["Win_percentage"] = n_res["Winners"] / n_res["Number of trades"]
    n_res["Average gain"] = np.array(gains).mean()
    n_res["Average lose"] = np.array(loses).mean()
    #n_res["Profit factor"] = (n_res['Win_percentage'] * n_res['Average gain']) \
    #    /((1 - n_res['Win_percentage'])* n_res['Average lose'])
    n_res["Average cost"]=np.array(costs).mean()
    n_res["Return"] = (n_res["PnL"] / n_res["Average cost"])*100
    n_res.set_index("Symbol",drop=True,inplace=True)
    
    # We add the variable calculated at the DataFrames which summarise the backtest results
    results = pd.concat([results,n_res])

    # If wanted we show some graphs
    if graphs:
        plt.plot(stock_data['Close'])
        plt.plot(stock_data['Position'] * stock_data['Close'].max())
        plt.show()
        plt.plot(stock_data['Total_pnl'])
        plt.show()

print(results)
print(f"Gain final :{results['PnL'].sum()}")
print(f"Win rate : {results['Win_percentage'].mean()}")
print(f"Average win: {results['Average gain'].mean()}")
print(f"Average lose: {results['Average lose'].mean()}")
print(f"Average return: {results['Return'].mean()}")