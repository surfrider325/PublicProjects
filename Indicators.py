import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
import os
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import statsmodels.formula.api as sm
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import yfinance as yf
from scipy.signal import argrelextrema
from collections import defaultdict
import bs4 as bs    
import pickle    
import requests    
import lxml

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')        
    soup = bs.BeautifulSoup(resp.text,'lxml')        
    table = soup.find('table', {'class': 'wikitable sortable'})        

    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    
    tickers = [re.sub('\n','',x) for x in tickers]

    return tickers    

def get_ticker(ticker,days):
    my_ticker = yf.Ticker(ticker)
    df = my_ticker.history(period="{}d".format(days), interval = "1h")
    df = df.reset_index()
    df['date'] = df.Datetime.dt.tz_localize(None)
    df.columns = [x.lower() for x in df.columns]
    
    return df

def get_sma(df,SMAs):
    sma = df.copy()
    for n in SMAs:
        sma['SMA{}'.format(n)] = sma['close'].rolling(window=n, center=False).mean()
    
    return sma
    
def get_max_min(prices, smoothing, window_range):
    smooth_prices = prices['close'].rolling(window=smoothing).mean().dropna()
    local_max = argrelextrema(smooth_prices.values, np.greater)[0]
    local_min = argrelextrema(smooth_prices.values, np.less)[0]
    price_local_max_dt = []
    for i in local_max:
        if (i>window_range) and (i<len(prices)-window_range):
            price_local_max_dt.append(prices.iloc[i-window_range:i+window_range]['close'].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i>window_range) and (i<len(prices)-window_range):
            price_local_min_dt.append(prices.iloc[i-window_range:i+window_range]['close'].idxmin())  
    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
    max_min = pd.concat([maxima, minima]).sort_index()
    #max_min.index.name = 'date'
    max_min = max_min.reset_index(drop=True)
    #max_min = max_min[~max_min.date.duplicated()]
    #p = prices.reset_index()   
    #max_min['day_num'] = p[p['timestamp'].isin(max_min.date)].index.values
    #max_min = max_min.set_index('day_num')['close']
    
    return max_min[['date','close']]

def find_IHS(max_min):  
    patterns = defaultdict(list)
    
    # Window range is 5 units
    for i in range(5, len(max_min)):  
        window = max_min.iloc[i-5:i]['close']
        
        # Pattern must play out in less than n units
        if window.index[-1] - window.index[0] > 100:      
            continue   
            
        a, b, c, d, e = window.iloc[0:5]
                
        # IHS
        if a<b and c<a and c<e and c<d and e<d and abs(b-d)<=np.mean([b,d])*0.02:
               patterns['IHS'].append((window.index[0], window.index[-1]))
        
    final = pd.DataFrame()
    for x,y in patterns['IHS']:
        inv_min = list(max_min[max_min.index==x]['date'])[0]
        inv_max = list(max_min[max_min.index==y]['date'])[0]
        final = pd.concat([final,pd.DataFrame({'start_event':[inv_min],'end_event':[inv_max]})])
        
    return final

def find_FW(max_min,buffer):  
    patterns = defaultdict(list)
    
    # Window range is 5 units
    for i in range(5, len(max_min)):  
        window = max_min.iloc[i-5:i]['close']
        
        # Pattern must play out in less than n units
        if window.index[-1] - window.index[0] > 100:      
            continue   
            
        a, b, c, d, e = window.iloc[0:5]
        
        m = (e-a)/4
        x1 = i-5
        intercept = a - x1*m
        def lower_line(x):
            y = m*x+intercept
            
            return y
            
        m = (d-b)/2
        x1 = i-3
        intercept = a - x1*m
        def upper_line(x):
            y = m*x+intercept
            
            return y
                
        # FW
        c1 = lower_line(i-3)
        e1 = lower_line(i)
        d1 = upper_line(i-1)
        if (c<a and a<d and d<b and e<d and e<c
        #if (a<b and c<a and c<d and d<b and e<d and e<c and f<d and g<d and
            #abs(c1-c)<=np.mean([c1,c])*buffer and abs(e1-e)<=np.mean([e1,e])*buffer and
            #and abs(d1-d)<=np.mean([d1,d])*buffer
           ):
               patterns['FW'].append((window.index[0], window.index[-1]))
        
    final = pd.DataFrame()
    for x,y in patterns['FW']:
        inv_min = list(max_min[max_min.index==x]['date'])[0]
        inv_max = list(max_min[max_min.index==y]['date'])[0]
        final = pd.concat([final,pd.DataFrame({'start_event':[inv_min],'end_event':[inv_max]})])
        
    return final