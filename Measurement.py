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
import sqlite3
from itertools import chain
import Indicators

def Measure_event(df,events,N=30):
    A = len(events)
    for event in events:
        kList = list(df[(df[event]==1)&(df[event].shift(-1)==0)].index)
        dfList = list(chain(*[range(i,i+N,1) for i in kList]))
        #viewList = list(chain(*[range(i,i+10,1) for i in kList]))+list(chain(*[range(i-1,i-10,-1) for i in kList]))
        df[event+'_after'] = np.where(df.index.isin(dfList), 1, 0)
        #df[event+'_final'] = np.where((df[event+'_after'].shift(1)==1)&(df[event+'_after']!=1),1,0)
        df[event+'_none'] = np.where((df[event] + df[event+'_after']) == 0, 1, 0)
        df[event + '_group'] = np.where(df[event].shift(1) != df[event], 1, 0)
        df[event + '_group'] = df.groupby(['ticker'])[event + '_group'].cumsum()
        df[event + '_group2'] = np.where(df[event+'_after'].shift(1) != df[event+'_after'], 1, 0)
        df[event + '_group2'] = df.groupby(['ticker'])[event + '_group2'].cumsum()
        df[event + '_start_time'] = df.groupby(['ticker',event + '_group'])['date'].transform('min')
        df[event + '_end_time'] = df.groupby(['ticker',event + '_group'])['date'].transform('max')
        df[event + '_start_time2'] = df.groupby(['ticker',event + '_group2'])['date'].transform('min')
        df[event + '_end_time2'] = df.groupby(['ticker',event + '_group2'])['date'].transform('max')
        
    cols = [x for x in df.columns if '_none' in x]
    df['no_events'] = np.where(df[cols].sum(axis=1, numeric_only=True)==A, 1, 0)
        
    return df

def get_changes(df,N=30):
    df['upper_max'] = df['high'].rolling(window=N, center=False).max()
    df['lower_min'] = df['low'].rolling(window=N, center=False).min()
    df['upper_chng'] = (df['upper_max'] - df['close'].shift(N))/df['close']
    df['lower_chng'] = (df['lower_min'] - df['close'].shift(N))/df['close']
        
    return df

def combine_events(tickers,events,SMAs,smoothing=10,window=10,N=80,K=500):
    final = pd.DataFrame()
    for ticker in tickers:
        try:
            df = Indicators.main(ticker,events,K,SMAs,smoothing,window)
            df = Measure_event(df,events,N)
            df = get_changes(df,N)
            final = pd.concat([df,final])
        except Exception as e: 
            print(e)
    return final

def get_totals(df3, events):

    final4 = pd.DataFrame()
    for event in events:
        if [val for key, val in events.items() if event in key][0] == 'bull':
            final1 = df3[df3[event]==1].groupby(['ticker',event, event + '_start_time', event + '_end_time']).upper_chng.describe().reset_index()
            final2 = df3[(df3[event+'_after']==1)&(df3['date']==df3[event+'_end_time2'])].groupby(['ticker',
                event+'_after', event+ '_start_time2',event + '_end_time2']).upper_chng.describe().reset_index()
            final3 = df3[df3['no_events']==1].groupby(['ticker']).upper_chng.describe().reset_index()
        else:
            final1 = df3[df3[event]==1].groupby(['ticker',event, event + '_start_time', event + '_end_time']).lower_chng.describe().reset_index()
            final2 = df3[(df3[event+'_after']==1)&(df3['date']==df3[event+'_end_time2'])].groupby(['ticker',
                event+'_after', event+ '_start_time2',event + '_end_time2']).lower_chng.describe().reset_index()
            final3 = df3[df3['no_events']==1].groupby(['ticker']).lower_chng.describe().reset_index()

        ##rename columsn and remove unneccesary ones
        final1.rename(columns={'count':'event_observations'},inplace=True)
        final2.rename(columns={'count':'after_event_observations', 'mean':'after_event_mean'},inplace=True)
        final3.rename(columns={'50%':'median'},inplace=True)
    
        conn = sqlite3.connect(':memory:')
        #write the tables
        final1.to_sql('f1', conn, index=False)
        final2.to_sql('f2', conn, index=False)
        final3.to_sql('f3', conn, index=False)
        qry = '''
        select  
            f1.ticker,
            {} as event,
            f1.event_observations,
            f1.{} as event_start_time,
            f1.{} as event_end_time,
            f2.after_event_observations,
            f2.after_event_mean,
            f2.{} as after_event_start_time,
            f2.{} as after_event_end_time,
            f3.count,
            f3.mean,
            f3.min,
            f3.median,
            f3.max
        from
            f1 left join f2 on
            f1.ticker = f2.ticker
            and f2.{} = f1.{}
        left join f3 on
            f3.ticker = f1.ticker      
        '''.format(event,event+'_start_time',event+'_end_time',
                   event+'_start_time2',event+'_end_time2',
                   event+'_start_time2',event+'_end_time'
                  #,event+'_start_time2',event+'_end_time'
                  )
        final = pd.read_sql_query(qry, conn)
        #final = final.merge(final3, on=['ticker'])
        final['event'] = event
        if [val for key, val in events.items() if event in key][0] == 'bull':
            final['Indicator'] = np.where(final['after_event_mean'] > final['median'], 1, 0)
        else:
            final['Indicator'] = np.where(final['after_event_mean'] < final['median'], 1, 0)
    
        final4 = pd.concat([final,final4],axis=0)
        
    return final4

def main(events= {'fw_event':'bull','rw_event':'bear'},
         SMAs = [30,60], smoothing = 7, window = 7, M = 80,K = 500, bound = 0.02):
    tickers = Indicators.save_sp500_tickers()
    df3 = combine_events(tickers,events,SMAs,smoothing,window,M,K)
    df3.reset_index(inplace=True,drop=True)
    final4 = get_totals(df3,events)
    final4.reset_index(inplace=True)
    #final4['50%-2'] = pd.Series(["{0:.2f}%".format(val * 100) for val in final4['50%-2']], index = final4.index)

    eventSuccess = final4[(abs(final4['after_event_mean']-final4['median']) > bound)&
        (final4['event_observations']> 10)&
        (pd.to_datetime(final4['after_event_end_time'])<(datetime.today()-timedelta(days=1)))].groupby('event')['Indicator'].mean(). \
        reset_index(). \
        rename(columns={'Indicator':'event_success'})
    stockSuccess = final4[(abs(final4['after_event_mean']-final4['median']) > bound)&
        (final4['event_observations']> 10)&
        (pd.to_datetime(final4['after_event_end_time'])<(datetime.today()-timedelta(days=1)))].groupby(['event','ticker']). \
        agg({'Indicator':'mean','after_event_observations':'sum'}). \
        reset_index(). \
        rename(columns={'Indicator':'stock_success','after_event_observations':'event_count'})
    totalSuccess = eventSuccess.merge(stockSuccess, on = ['event'])
    totalSuccess = totalSuccess[['event','ticker','event_count','event_success','stock_success']]

    final4 = final4.merge(totalSuccess, on = ['event','ticker'])
    
    return df3, final4

if __name__ == '__main__':
    main()
