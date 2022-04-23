#https://github.com/mattpodolak/pmaw/blob/master/examples/search_comments.ipynb

import pandas as pd
import praw
from pmaw import PushshiftAPI

# instantiate
api = PushshiftAPI()

reddit = praw.Reddit(
 client_id='protected replace when running',
 client_secret='protected replace when running',
 user_agent=f'python: PMAW request enrichment (by u/Nursing Sentiment)'
)

api_praw = PushshiftAPI(praw=reddit)
posts = api.search_submissions(subreddit="nursing", limit=700000, before=1640998861, after=1609462861, mem_safe=True, safe_exit=True)
print(f'{len(posts)} posts retrieved from Pushshift')


# get all responses
post_list = [post for post in posts]

# convert submissions to dataframe
new_posts_df = pd.DataFrame(post_list)

#save dataframe to csv
#new_posts_df.to_csv('/Volumes/USBPM951/RedditNursing/nursingpmaw.csv', header=True, encoding='utf-8', index=False)

# store the extracted comments into a csv file for later use
new_posts_df.to_csv('/Volumes/USBPM951/RedditNursing/nursingpmaw.csv', header=True, index=False, columns=list(new_posts_df.axes[1]))
