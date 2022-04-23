import pandas as pd
df = pd.read_csv('/Volumes/USBPM951/RedditNursing/nursingraw2.csv')
#df = pd.read_csv('/Volumes/USBPM951/RedditNursing/nursingraw2.csv', dtype={'created_utc': int})
df.head()
df.shape
#Add in new time variable
import datetime as dt
#created_date = dt.datetime.fromtimestamp(df.created_utc)
#df['intcreated_utc'] = df['created_utc'].astype(int)

#Drop post that were removed from Reddit
import numpy as np
df = df[np.logical_not(df['removed_by_category'].isin(['reddit', 'automod_filtered', 'moderator', 'deleted']))]
df.shape
df.info()
#df.to_csv('/Volumes/USBPM951/RedditNursing/nurcleaned.csv', encoding='utf-8', index=False)

#Import additional libraries
# libraries for sentiment analysis (polarity)
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, RegexpTokenizer # tokenize words
from nltk.corpus import stopwords

# libraries visualization
#import matplotlib.pyplot as plt
#%matplotlib inline
#plt.rcParams["figure.figsize"] = (10, 8) # default plot size
#import seaborn as sns
#sns.set(style='whitegrid', palette='Dark2')
import numpy as np

#misc libraries
from pprint import pprint
from itertools import chain
#from wordcloud import WordCloud

#vader_lexicon: Dataset of lexicons containing the sentiments of
# specific texts which powers the Vader Sentiment Analysis
#punkt: Pre-trained models that help us tokenize sentences.
#stopwords: Dataset of common stopwords in English
nltk.download('vader_lexicon') # get lexicons data
nltk.download('punkt') # for tokenizer
nltk.download('stopwords')
#Clean selftext fill to convert nan to blank for processing
df['selftextc'] = df.selftext.fillna('')
df.info()

#Run analyser to get scores on cleaned selftextc
sid = SentimentIntensityAnalyzer()
res = [*df['selftextc'].apply(sid.polarity_scores)]
pprint(res[:3])

#With the scores calculated in dictionaries, we create a data frame using from_records
# and then concatenate it to our data frame on an inner join.
sentiment_df = pd.DataFrame.from_records(res)
df = pd.concat([df, sentiment_df], axis=1, join='inner')
df.head()
df.info()
#Save to excel
#df.to_excel('/Users/renitawashburn/PycharmProjects/RedditNursing2022/nursing_output1a.xlsx', index=False)

#Now that we have the polarity scores, the next step is to choose a threshold to label the text as
# positive, negative, or neutral.  Typical threshold is .05.

THRESHOLD = 0.05

conditions = [
    (df['compound'] <= -THRESHOLD),
    (df['compound'] > -THRESHOLD) & (df['compound'] < THRESHOLD),
    (df['compound'] >= THRESHOLD),
    ]

values = ["neg", "neu", "pos"]
df['label'] = np.select(conditions, values)

df.head()

#Save to excel
#df.to_csv('/Volumes/USBPM951/RedditNursing/nursing_output2a.csv', index = False)


#Text2emotion is the python package developed with the clear intension to find the appropriate emotions embedded
#in the text data. 5 basic emotion categories such as Happy, Angry, Sad, Surprise, and Fear
#Higher the score of a particular emotion category, we can conclude that the message belongs to that category.
#Import the modules
import text2emotion as te
import nltk
#nltk.download()

# assign an emotion to each selftextc
df['emotion'] = df.selftextc.apply(lambda x: te.get_emotion(x))
# exploding the dictionary into 4 different columns, based on the dictionary keys
df = pd.concat([df, pd.DataFrame(df['emotion'].tolist())], axis =1)

#Call to the function
#text = df['selftextc']
#emo = te.get_emotion(text)
#pprint(emo[:5])

#With the scores calculated in dictionaries, we create a data frame using from_records
# and then concatenate it to our data frame on an inner join.
#emotion_df = pd.DataFrame.from_records(emo)

#df = pd.concat([df, emotion_df], axis=1, join='inner')
#df.info()

#Save to excel
df.to_csv('/Volumes/USBPM951/RedditNursing/nursingsent.csv', encoding='utf-8', index=False)





