#!/usr/bin/env python
# coding: utf-8




#For Pre-Processing
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer 
from nltk.tokenize import word_tokenize
import os, re, csv, math, codecs

# For Training
import keras
from keras import optimizers
from keras import backend as K
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.utils import plot_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping

# For array, dataset, and visualizing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
np.random.seed(0)
MAX_NB_WORDS = 180000




#Reading the crawled Twitter .csv file
tw=pd.read_csv("~/sanjibwork/SRTwitter_Topic_Classification_dataset.csv")






tw.head(15)





nltk.download('stopwords')





nltk.download('punkt')





#preprocessing of tweets with Regex

from nltk import word_tokenize
def preprocess_tweet(text):

      
    # convert text to lower-case
    nopunc = text.lower()
    
    # remove URLs
    nopunc = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', nopunc)
    nopunc = re.sub(r'http\S+', '', nopunc)
    # remove usernames
    nopunc = re.sub('@[^\s]+', '', nopunc)
    
    #remove 'rt'
    nopunc = re.sub(r'rt([^\w]+)', '', nopunc)
       
    return nopunc





tw['clean_text'] = [preprocess_tweet(w) for w in tw['text'].tolist()]
tw['clean_text'] = tw['clean_text'].str.replace("[^a-zA-Z]", " ")
tweet_list = tw['clean_text'].astype(str).tolist()





from nltk.corpus import stopwords
processed_list = []
c=0
for sentence in tqdm(tweet_list):
    text_tokens = word_tokenize(sentence)
    filtered = [word for word in text_tokens if word not in stopwords.words('english')]
    processed_list.append(" ".join(filtered))
    c+=1





tw["clean_text"] = processed_list





df = pd.DataFrame()
df["text"] = tw["clean_text"]
df["label"]=tw["label"]
df=df.dropna(inplace=False,how='any')
df[pd.isna(df['text'])]['text']="empty"




#Processed Twitter .csv file after processing 
df.to_csv("Process_withoutnan_SRTwitter_Topic_Classification_dataset.csv", index=False)







len(df)





df.head(5)





df['label'].value_counts()

