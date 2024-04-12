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

def chart_all(df,events,events_df,SMAs,minmax):
    ticker = list(df.ticker.unique())[0]
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price')])
    for n in SMAs:
        rgb = 'rgb({},{},{})'.format(250-n*2,n*2,250-n*2)
        fig.add_trace(go.Scatter(x=df['date'],
                y=df['SMA{}'.format(n)],
                mode='lines',
                name='SMA{}'.format(n),
                line=dict(color=rgb)))
    fig.add_trace(go.Scatter(x=minmax['date'],
                y=minmax['close'],
                mode='markers',
                name='Local Min/Max',
                marker=dict(color='rgb(200,200,0)',
                            size=10,
                            opacity=0.2)
                        ))
    # iterate through the shaded regions dataframe
    for event in events:
        for index, row in events_df[event].iterrows():
    
            # retrieve the dates
            start = row['start_event']
            end = row['end_event']

            # add shaded region
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor="grey",
                opacity=0.1,
                line_width=1,
                annotation_text=event,
                annotation_textangle=-45
            )   
 
    fig.update_xaxes(
        rangeslider_visible=True,
        rangebreaks=[
            # NOTE: Below values are bound (not single values), ie. hide x to y
            dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
            dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
            #dict(values=["2024-12-25", "2024-01-01","2024-01-15","2024-02-19"])  # hide holidays (Christmas and New Year's, etc)
            ]
    )
    fig.update_layout(title={'text':ticker, 'xanchor':'center', 'yanchor':'top','x':0.5},
                height=600,
                title_font={"family":"arial","color":"gray","size":38},
                paper_bgcolor='rgb(220,230,230)'
                 )
    fig.show()
    
def chart_event(df3, event, ticker, SMAs):
    df = df3[df3.ticker==ticker].copy()
    fig = go.Figure()
    fig = go.Figure(data=[go.Candlestick(x=df['date'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price')])
    for n in SMAs:
        rgb = 'rgb({},{},{})'.format(250-n*2,n*2,250-n*2)
        fig.add_trace(go.Scatter(x=df['date'],
                y=df['SMA{}'.format(n)],
                mode='lines',
                name='SMA{}'.format(n),
                line=dict(color=rgb)))
    
    # iterate through the shaded regions dataframe
    df[event] = np.where(df[event + '_after']==1,-1,df[event])
    df[event+'_start_time'] = np.where(df[event + '_after']==1, df[event+'_start_time2'],df[event+'_start_time'])
    df[event+'_end_time'] = np.where(df[event + '_after']==1, df[event+'_end_time2'],df[event+'_end_time'])
    rw = df.groupby([event+'_start_time',event+'_end_time'])[event].mean().reset_index()
    rw = rw[rw[event].isin([1,-1])].reset_index(drop=True)
    eventName = {1:'During {}'.format(event),-1:'After {}'.format(event),0:'No {}'.format(event)}
    eventList = list(rw[event])
    eventList = [eventName.get(item,item)  for item in eventList]
    for index, row in rw.iterrows():
        eventValue = eventList[index]
        # retrieve the dates
        if re.match('During .*',eventValue):
            fcolor = "grey"
        else:
            fcolor = "blue"
        start = row[event+'_start_time']
        end = row[event+'_end_time']
        # add shaded region
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=fcolor,
            opacity=0.1,
            line_width=1,
            annotation_text=eventValue,
            annotation_textangle=-45
        )  
    fig.update_xaxes(
        rangeslider_visible=True,
        rangebreaks=[
            # NOTE: Below values are bound (not single values), ie. hide x to y
            dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
            dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
            #dict(values=["2024-12-25", "2024-01-01","2024-01-15","2024-02-19"])  # hide holidays (Christmas and New Year's, etc)
        ]
    )
    fig.update_layout(title={'text':ticker, 'xanchor':'center', 'yanchor':'top','x':0.5},
                height=600,
                title_font={"family":"arial","color":"gray","size":38},
                paper_bgcolor='rgb(220,230,230)'
                 )
    fig.show()
    
