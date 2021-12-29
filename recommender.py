
# BUILDING A SIMPLE MOVIE RECOMMENDER
# LOADING THE DATA

import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')

data = pd.read_csv('C:/ratings.csv')
data.head(10)

movie_titles_genre = pd.read_csv("C:/movies.csv")
movie_titles_genre.head(10)

data = data.merge(movie_titles_genre, on='movieId', how='left')
data.head(10)

# FEATURE ENGINEERING
# AVERAGE RATING

Average_ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
Average_ratings.head(10)

# TOTAL NUMBER OF RATING

Average_ratings['Total Ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())
Average_ratings.head(10)

# BUILDING THE RECOMMENDER
# CALCULATING THE CORRELATION

movie_user = data.pivot_table(index='userId', columns='title', values='rating')

user_selected_movie = 'Incredibles, The (2004)'

movie_user = movie_user[movie_user.get(user_selected_movie).notnull()]
movie_user = movie_user.dropna(axis='columns',thresh=2)

correlations = movie_user.corrwith(movie_user[user_selected_movie])
correlations.head()

recommendation = pd.DataFrame(correlations,columns=['Correlation'])
recommendation.dropna(inplace=True)

recommendation = recommendation.join(Average_ratings['Total Ratings'])
recommendation.head()

# TESTING THE RECOMMENDATION SYSTEM

recc = recommendation[recommendation['Total Ratings'] > 100].sort_values('Correlation', ascending=False).reset_index()

recc = recc.merge(movie_titles_genre, on='title', how='left')
recc.head(10)
