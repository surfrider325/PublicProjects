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

def Measure_event(df,events,N=30):
    for event in events:
        kList = list(df[(df[event]==1)&(df[event].shift(-1)==0)].index)
        dfList = list(chain(*[range(i+1,i+N,1) for i in kList]))
        #viewList = list(chain(*[range(i,i+10,1) for i in kList]))+list(chain(*[range(i-1,i-10,-1) for i in kList]))
        df[event] = np.where(df.index.isin(dfList),-1,df[event])
        
        df[event] = np.where((df[event].shift(1)==-1)&(df[event]!=-1),-2,df[event])
        
        
    return df

def get_changes(df,N=30):
    df['upper_max'] = df['high'].rolling(window=N, center=False).max()
    df['lower_min'] = df['low'].rolling(window=N, center=False).min()
    df['upper_chng'] = (df['upper_max'] - df['close'].shift(N))/df['close']
    df['lower_chng'] = (df['lower_min'] - df['close'].shift(N))/df['close']
        
    return df


