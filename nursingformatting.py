#Description: File to perform a few formatting task.  Correct Emotion breakouts
#Calculate percent change for COVID variables

import pandas as pd
col_list = ['score', 'title', 'selftext', 'created_utc', 'id', 'removed_by_category',
            'author', 'num_comments', 'over_18', 'newdate_x', 'created_date',
            'link_date', 'day_of_week',	'hour', 'st_count', 'Date', 'prevday_admad_covid', 'tot_adpt_covid',
            'deaths_covid', 'prevday_admped_covid', 'tot_pedpt_covid', 'prevday_adm_all', 'tot_pthosp', 'tot_prevday_adm_covid',
            'selftextc', 'neg', 'neu', 'pos', 'compound', 'label', 'emotion']

df = pd.read_csv('/Volumes/USBPM951/RedditNursing/nursingsent.csv', usecols=col_list)
df.head(10)
df.info()

#Correct a few columns
##Expand Emotion Scores
#df['happy'] = emotion.split(", ")
import re
df['happy'], df['angry'], df['suprise'], df['sad'], df['fear'] = df['emotion'].str.split(',', 4).str
#Keep only numerics from split cells

#df['happy'] = re.sub("[^0-9]", "", df['happy'])
df['happy'] = (df['happy'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int))/100
df['angry'] = (df['angry'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int))/100
df['suprise'] = (df['suprise'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int))/100
df['sad'] = (df['sad'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int))/100
df['fear'] = (df['fear'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int))/100
df.head(10)

#Export file to csv
df.to_csv('/Volumes/USBPM951/RedditNursing/nursingexp.csv', encoding='utf-8', index=False)


#Import COVID Hospitilizaiton Data 2021
#import pandas as pd
col_list2 = ['Date','prevday_admad_covid', 'tot_adpt_covid', 'deaths_covid', 'prevday_admped_covid',
    'tot_pedpt_covid', 'prevday_adm_all','tot_pthosp']
df2 = pd.read_csv('/Users/renitawashburn/PycharmProjects/RedditNursing/COVID-19_Reported_2021.csv', usecols=col_list2)

#Create data variable to link to Reddit data file
df2['intcreated_utc'] = pd.to_datetime(df2['Date'])
#df2['intcreated_utc'] = df2['Date'].astype(float(int))
df2['newdate'] = pd.to_datetime(df2['intcreated_utc'], unit='s')
df2['link_date'] = df2['newdate'].dt.strftime('%-m-%d-%Y')


#Create percent change variables for COVID Data
#pct_change() method calculates the percentage change only between the rows of data
#calculate percent change between consecutive values in 'sales' column
df2.sort_values('link_date')

#df['sales_pct_change'] = df['sales'].pct_change()
df2['pc_tot_adpt_covid'] = df2['tot_adpt_covid'].pct_change()
df2['pc_tot_pedpt_covid'] = df2['tot_pedpt_covid'].pct_change()
df2['pc_tot_pthosp'] = df2['tot_pthosp'].pct_change()
df2['pc_prevday_adm_all'] = df2['prevday_adm_all'].pct_change()
df2.head(10)
#Export COVID reference file to csv
df2.to_csv('/Volumes/USBPM951/RedditNursing/covidref.csv', encoding='utf-8', index=False)
#Import COVID ref to join with nursingsent data
df3 = pd.read_csv('/Volumes/USBPM951/RedditNursing/covidref.csv')
df3.head()
df3.info()

#Combine new columns to nursingsent data
df_cd = pd.merge(df, df3, how='left', left_on='link_date', right_on='link_date')
df_cd.head(10)

#Export combined file to csv
df_cd.to_csv('/Volumes/USBPM951/RedditNursing/nursingML.csv', encoding='utf-8', index=False)