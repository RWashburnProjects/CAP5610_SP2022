#CAP5610 Capstone Project
#Feature Creation Program (

import pandas as pd
import time
from datetime import datetime, timezone


#Import Reddit r/nursing file
#Only keep columns of interest
col_list = ['score', 'title', 'selftext', 'created_utc', 'id', 'removed_by_category',
            'author', 'num_comments', 'over_18']
df = pd.read_csv('/Volumes/USBPM951/RedditNursing/nursingpmaw.csv', usecols=col_list)
df.head(10)
df.info()

#Convert Reddit epoch date to human readable format
import datetime

df['intcreated_utc'] = df['created_utc'].astype(int)
df['newdate'] = pd.to_datetime(df['intcreated_utc'], unit='s')
df['created_date'] = df['newdate'].dt.strftime('%Y-%m-%d-%H:%M')
df['link_date'] = df['newdate'].dt.strftime('%-m-%d-%Y')
#df['newdate'] = pd.to_datetime(df['intcreated_utc'], format='%Y-%m-%d %H:%M:%S-%Z', errors='coerce')
df.info()
df.head(10)
#Create day of the week and hours variables

df['created_date'] = pd.to_datetime(df['created_date'])
df['day_of_week'] = df['created_date'].dt.day_name()
df['hour'] = pd.to_datetime(df['created_date'].astype(str)).dt.hour
#df.info

#Create count of words in selftext & title
df['st_count'] = df['selftext'].str.count(' ') + 1
df['tit_count'] = df['title'].str.count(' ') + 1
df.info

#Import COVID Hospitilizaiton Data 2021
#import pandas as pd
col_list2 = ['Date','prevday_admad_covid', 'tot_adpt_covid', 'deaths_covid', 'prevday_admped_covid',
    'tot_pedpt_covid', 'prevday_adm_all','tot_pthosp']
df2 = pd.read_csv('/Users/renitawashburn/PycharmProjects/RedditNursing/COVID-19_Reported_2021.csv', usecols=col_list2)

df2.info()
#Create data variable to link to Reddit data file
df2['intcreated_utc'] = pd.to_datetime(df2['Date'])
#df2['intcreated_utc'] = df2['Date'].astype(float(int))
df2['newdate'] = pd.to_datetime(df2['intcreated_utc'], unit='s')
df2['link_date'] = df2['newdate'].dt.strftime('%-m-%d-%Y')
df2.info
#Create percent change variables for COVID Data
#pct_change() method calculates the percentage change only between the rows of data
#calculate percent change between consecutive values in 'sales' column
df2.sort_values('link_date')



#df['sales_pct_change'] = df['sales'].pct_change()
df2['pc_tot_adpt_covid'] = df2['tot_adpt_covid'].pct_change()
df2['pc_tot_pedpt_covid'] = df2['tot_pedpt_covid'].pct_change()
df2['pc_tot_pthosp'] = df2['tot_pthosp'].pct_change()
df2['pc_prevday_adm_all'] = df2['prevday_adm_all'].pct_change()

#df2['pc_tot_adpt_covid'] = df2.groupby(['Date'])['tot_adpt_covid'].pct_change()
#df2['pc_tot_pedpt_covid'] = df2.groupby(['Date'])['tot_pedpt_covid'].pct_change()
#df2['pc_tot_pthosp'] = df2.groupby(['Date'])['tot_pthosp'].pct_change()
#df2['pc_prevday_adm_all'] = df2.groupby(['Date'])['prevday_adm_all'].pct_change()

#Create total previous day COVID admission
sum_prevadm = df2["prevday_admad_covid"] + df2["prevday_admped_covid"]
df2['tot_prevday_adm_covid'] = sum_prevadm
#df.pct_change(freq='5D')

#Combine Reddit and COVID Data
df_cd = pd.merge(df, df2, how='left', left_on = 'link_date', right_on = 'link_date')
df_cd.info

#Export combined file to csv
df_cd.to_csv('/Volumes/USBPM951/RedditNursing/nursingraw2.csv', encoding='utf-8', index=False)