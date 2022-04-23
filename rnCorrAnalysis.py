import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Read in CSV
df = pd.read_csv('/Volumes/USBPM951/MLeditedemotionV2.csv')
df.shape

#Check distibution of num_comments to determine filter
#df.hist(column="num_comments", bins=10, figsize=(12,8), range=[0, 20])

#Drop rows were selftext is nan. Drop removed by category column. Drop num comments less than 1
df = df[df['selftext'].notna()]
df = df.drop('removed_by_category', 1)
df = df.loc[df["num_comments"] > 5]
df.shape

#Feature creation
import re
#Title word count
#wc_title = len(re.findall(r'\w+', 'title'))
wc_title = df['title'].str.lower().str.split()
wc_title2 = wc_title.apply(len)
wc_selftext = df['selftextc'].str.lower().str.split()
wc_selftext2 = wc_selftext.apply(len)

#res = len(test_string.split())
#Self text word count
#wc_selftext= len(re.findall(r'\w+', 'selftextc'))
df['wc_title2']  = wc_title2
df['wc_selftext2'] = wc_selftext2

df2 = df.drop(['emotion', 'selftextc','author','created_utc','id','selftext','title','newdate_x',
                     'created_date','link_date','Date_x','day_of_week', 'KEY'], axis=1)

from scipy.stats import chi2_contingency

def cramers_V(var1,var2) :
  crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) # Cross table building
  stat = chi2_contingency(crosstab)[0] # Keeping of the test statistic of the Chi2 test
  obs = np.sum(crosstab) # Number of observations
  mini = min(crosstab.shape)-1 # Take the minimum value between the columns and the rows of the cross table
  return (stat/(obs*mini))


rows = []

for var1 in df2:
    col = []
    for var2 in df2:
        cramers = cramers_V(df2[var1], df2[var2])  # Cramer's V test
        col.append(round(cramers, 2))  # Keeping of the rounded value of the Cramer's V
    rows.append(col)

cramers_results = np.array(rows)
df3 = pd.DataFrame(cramers_results, columns=df2.columns, index=df2.columns)
df3



from dython.nominal import associations
# Plot features associations
plt.rcParams.update({'font.size': 6})
plt.rcParams["figure.autolayout"] = True
plt.legend(prop={'size': 6})
theils_results=associations(df2, nom_nom_assoc='theil', figsize=(10, 10))

#df4 = pd.DataFrame(associations(df2, nom_nom_assoc='theil'), columns=df2.columns, index=df2.columns)
