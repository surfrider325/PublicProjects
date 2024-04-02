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
        dfList = list(chain(*[range(i+1,i+N,1) for i in kList]))
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
            df = Indicators.main(ticker,K,SMAs,smoothing,window)
            df = Measure_event(df,events,N)
            df = get_changes(df,N)
            final = pd.concat([df,final])
        except Exception as e: 
            print(e)
    return final


