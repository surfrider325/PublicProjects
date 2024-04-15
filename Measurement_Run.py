from prefect import flow
from prefect.blocks.notifications import SlackWebhook
from tabulate import tabulate

import Indicators
import Measurement
import Charts

slack_webhook_block = SlackWebhook.load("slack")

M = 80
K = 500
window = 7
smoothing = 7
events = {'ihs_event':'bull','hs_event':'bear','fw_event':'bull','rw_event':'bear'}
bound = 0.03
SMAs = [30,60,90]

@flow(log_prints=True)
def Measurement_Run():
    df3, final4 = Measurement.main(events, SMAs, smoothing, window, M, K, bound)
    df_print = final4.sort_values(['event_end_time'],ascending=0).head(15)
    df_print.drop(['index','after_event_observations','after_event_mean',
                   'after_event_start_time','after_event_end_time',
                   'count','mean','min','median','max','Indicator'],axis=1,inplace=True)
    
    slack_webhook_block.notify("```\n" + print(tabulate(df_print, headers='keys', tablefmt="grid")) + "\n```")
    
if __name__ == "__main__":
    Measurement_Run()