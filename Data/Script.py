import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense, LSTM
from keras import regularizers
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dropout
import talos
import matplotlib.pyplot as plt

import praw
from psaw import PushshiftAPI
import config
import pandas as pd

import nltk
from nltk.corpus import stopwords
import string
from  nltk import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.collocations import *
# nltk.download('wordnet')
from nltk import word_tokenize, FreqDist

from spacy import displacy
import spacy
from tqdm import tqdm

tqdm.pandas()
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('spacytextblob')

import plotly.graph_objects as go
import plotly.express as px

import twint
import pandas as pd

import nest_asyncio 
nest_asyncio.apply()

from dateutil import rrule
import datetime as datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import time
start_time = time.time()

r = praw.Reddit(
    client_id=config.reddit['client_id'],
    client_secret=config.reddit['client_secret'],
    username=config.reddit['username'],
    password=config.reddit['password'],
    user_agent='test'
)
api = PushshiftAPI(r)

currentDate = datetime.datetime.now()
print('Enter number of months back you want to look at:')
num_months = input()
print('Enter ticker symbol for the stock:')
ticker = input()#ticker symbol
print('Enter subreddit you want to webscrape from')
sub = input()#Subreddit
print('Enter limit of post you want to scrape from reddit')
limit = input()# limit of red posts




def populate(currentDate, num_months, ticker, subreddit, limit): #num_months = number of months back you want to look at. ticker = stock ticker symbol. subreddit = the reddit subreddit you want to scrape from. limit is the limit number of post you want to scrape from reddit.
#Reddit Posts    
    Dates = []
    gen = []

    for i in range(len(num_months)):#Number of months
        Dates.append((currentDate - relativedelta(months=i)).date())
    
    for i in range(len(Dates)-1):

        
        gen.extend(list(api.search_submissions(after=Dates[i+1],
                            before=Dates[i],
                            q=ticker, 
                            subreddit='wallstreetbets',
                            filter=['created', 'title'],
                            check_for_async=False,
                            limit=len(limit))))
    
#Turning gen list into a pandas dataframe    
    reddit_posts = []

    for i in gen:
        reddit_posts.append([i.title, 
#               i.author, 
#                   i.subreddit, 
#               i.score, 
                datetime.datetime.fromtimestamp(
                int(round(i.created))
                ).strftime('%Y-%m-%d')])#Returns the time the post was created in regular time instead of UNIX time])
        
    red_posts = pd.DataFrame(reddit_posts,columns=['post', 
#                                 'author', 
#                                 'subreddit', 
#                                 'score', 
                                'date'])
    
    
#Tweets dataframe
    
    c = twint.Config()

    c.Search =ticker #search must contain ""
    c.Min_likes = 500 #min number of likes
    c.Until = str(currentDate.date()) #return tweets that were published before this date
    c.Since = str((currentDate - relativedelta(months=len(num_months))).date()) #return tweets published after this day(Months)
    c.Count = True
    c.Limit = 150000 # Limits the tweets to this number
    # c.Format = "Tweet id: {id} | Date: {date} | Username: {username} | Tweet: {tweet} | Mention: {mention}"
    c.Store_csv = True
    c.Output = 'Post.csv'
    
    twint.run.Search(c)
    
    tweets = pd.read_csv('Post.csv')
    tweets = tweets.rename(columns={"tweet": "post"})

    
    tweets = pd.concat([tweets['date'], tweets['post']], axis=1)
    
    
#combining reddit post and tweets    
    posts = pd.concat([red_posts, tweets])
#Dropping duplicate posts
    posts = posts.drop_duplicates(subset=['post'], ignore_index=True)
    
    
#NLP
#Add a column of the tokenized posts
    posts['spacy'] = posts.post.progress_apply(lambda x: nlp(x))
#Adding a sentiment column\    
    posts['sentiment'] = posts.spacy.progress_apply(lambda x: (x._.polarity))
#Adding a subjectivity column 
    posts['subjectivity'] = posts.spacy.progress_apply(lambda x: (x._.subjectivity))
#grouping the sentiment and subjectiity means by date
    final_posts = posts.groupby(['date']).mean().sort_values(by='date', ascending=False)
#capitalizing index    
    final_posts.index.names = ['Date']
#webscraping stock price data on ticker symbol
    tick = web.DataReader(ticker, data_source='yahoo', start=str((currentDate - relativedelta(months=len(num_months))).date()), end=str(currentDate.date()))
#merging post sentiment and stock price
    df= tick.join(final_posts)
#Filling Nan with the column mean
    df = df.fillna(df.mean())
#creating a target column with tomorrows price and droping todays date.    
    df = pd.DataFrame(df)
    df['Target'] = np.append(df['Close'].iloc[1:].values, [np.nan])
    df = df.dropna()
    
#Train-test Split    
    X = df.filter(['Close', 'sentiment', 'subjectivity'])
    y = df['Target']


    training_data_len = math.ceil(len(X) * .8)

    X_train = X.iloc[:training_data_len]
    y_train = y.iloc[:training_data_len]
    X_test = X.iloc[training_data_len -10:]
    y_test = y.iloc[training_data_len:]
    
    
#Scaling data
    sx = MinMaxScaler(feature_range=(0,1))
    sy = MinMaxScaler(feature_range=(0,1))
    
    scaled_X_train = sx.fit_transform(X_train)
    scaled_X_test = sx.transform(X_test)
    scaled_y_train = sy.fit_transform(np.array(y_train).reshape(-1, 1))
    scaled_y_test = sy.transform(np.array(y_test).reshape(-1, 1))
    
#setting up the train set    
#each Timestep uses window to predict the next value
    window = 10
    X = []
    y = []

    for i in range(window, len(X_train)):
        X.append(scaled_X_train[i-window:i,:])
        y.append(scaled_y_train[i])
        
#turning X, y into arrays
    X, y = np.array(X), np.array(y)
    
#LSTM expects the data to be 3 dimensional. in order of number of samples, number of timesteps, and number of features
    X = np.reshape(X, (X.shape[0], X.shape[1], 3))
    
#setting up the test set. just like the the training set, the test set needs to use the last 10 days to predict tomorrows value
    window = 10
    Xt = []
    yt = scaled_y_test

    for i in range(window, len(X_test)):
        Xt.append(scaled_X_test[i-window:i,:])
        
#turning Xt, yt into arrays
    Xt, yt = np.array(Xt), np.array(yt)
#changing the shape of the Xt array
    Xt = np.reshape(Xt, (Xt.shape[0], Xt.shape[1], 3))
    
    
#Ceating the LSTM
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape= (X.shape[1], 3)))
    model.add(Dense(50))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(25))
    model.add(Dense(1))
    
#Compliling the model
    model.compile(optimizer='adam', 
               loss='mean_squared_error',
               metrics = ["mean_absolute_percentage_error"])

    history = model.fit(X, y, 
                        batch_size=20,
                        epochs=100,
                        validation_data=(Xt, yt))
#getting predicted price values
    predictions = model.predict(Xt)
    predictions = sy.inverse_transform(predictions)
    yt_unscaled = sy.inverse_transform(yt)
#Checking rmse
    rmse = np.sqrt(((predictions - yt_unscaled) ** 2).mean())
#results
    results_train = model.evaluate(X, y)
    print(f'Training Loss: {results_train[0]:.3} \nTraining MAPE: {results_train[1]:.3}')

    print('----------')

    results_test = model.evaluate(Xt, yt)
    print(f'Test Loss: {results_test[0]:.3} \nTest MAPE: {results_test[1]:.3}')
    print('RMSE:', rmse)



    
    return print(f'Training Loss: {results_train[0]:.3} \nTraining MAPE: {results_train[1]:.3}'), print(f'Test Loss: {results_test[0]:.3} \nTest MAPE: {results_test[1]:.3}'), print('RMSE:', rmse)

populate(currentDate, num_months, ticker, sub, limit)