#!/usr/bin/env python
# coding: utf-8

# # Building A Simple Movie Recommender

# # Loading the data

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv('C:/ratings.csv')
data.head(10)


# In[3]:


movie_titles_genre = pd.read_csv("C:/movies.csv")
movie_titles_genre.head(10)


# ### Merging 

# In[4]:


data = data.merge(movie_titles_genre, on='movieId', how='left')
data.head(10)


# # Feature Engineering

# ## Average Rating

# In[5]:


Average_ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
Average_ratings.head(10)


# ## Totel Number of Ratings

# In[6]:


Average_ratings['Total Ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())
Average_ratings.head(10)


# # Building The Recommender

# ## Calculating The Correlation

# In[7]:


movie_user = data.pivot_table(index='userId', columns='title', values='rating')
movie_user.head(10)

'''
   Now we need to select a movie to test our recommender system.
   Choose any movie title from the data. Here, I chose Toy Story (1995)
'''
# In[8]:


user_selected_movie = 'Toy Story (1995)'


# In[9]:


movie_user = movie_user[movie_user.get(user_selected_movie).notnull()]
movie_user = movie_user.dropna(axis='columns', thresh=2)

correlations = movie_user.corrwith(movie_user[user_selected_movie])
correlations.head()

recommendation = pd.DataFrame(correlations, columns=['Correlation'])
recommendation.dropna(inplace=True)

recommendation = recommendation.join(Average_ratings['Total Ratings'])
recommendation.head()


# # Testing The Recommendation System

# In[10]:


recc = recommendation[recommendation['Total Ratings'] > 100].sort_values('Correlation', ascending=False).reset_index()

recc = recc.merge(movie_titles_genre, on='title', how='left')
recc.head(10)


# # Result
'''
  We can see that the top recommendations are pretty good. 
  The movie that has the highest/full correlation to Toy Story is Toy Story itself. 
  The movies such as The Incredibles, Finding Nemo and Alladin show high correlation with Toy Story.
'''
