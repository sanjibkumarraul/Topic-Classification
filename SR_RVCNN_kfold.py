#!/usr/bin/env python
# coding: utf-8

# In[1]:


#d=pd.read_csv("~/Desktop/SR_Github_link/Process_withoutnan_SRTwitter_Topic_Classification_dataset.csv", keep_default_na=False)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import codecs, os
from sklearn.model_selection import cross_validate
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics

# For array, dataset, and visualizing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
np.random.seed(0)

nltk.download('stopwords')
nltk.download('punkt')
#MAX_NB_WORDS = 180000
import tensorflow as tf
from sklearn.utils import class_weight
# from sklearn import cross_validation 
from sklearn.model_selection import cross_validate
#from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from statistics import mean
from statistics import stdev


# In[2]:


#Reading a large pre trained fastext word embedding file(2.10GB)
#loading fastText model trained with pretrained_vectors
#finding word vectors from pretrained fasttext word embedding

embeddings_index = {}
f = codecs.open('./wiki-news-300d-1M.vec', encoding='utf-8')
#f = codecs.open('wiki-news-300d-1M.vec', encoding='latin-1')
for line in tqdm(f):
	values = line.rstrip().rsplit(' ')
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()


# In[3]:


tweets_df = pd.read_csv('~/Desktop/SR_Github_link/Process_withoutnan_SRTwitter_Topic_Classification_dataset.csv',header=0,encoding='latin-1',keep_default_na=False)


# In[45]:


tweets_df['label'].value_counts()


# In[4]:


#For testing different labelencoder format
encoder = preprocessing.LabelEncoder()
tweets_df['label_class']=encoder.fit_transform(tweets_df['label'])
tweets_df.tail()

tweet_list = tweets_df['text'].astype(str).tolist()
label_list1 = tweets_df['label'].tolist()   ## converting to list like [accident, business, cricket, education..]

label_list=tweets_df['label_class']
one_hot_labels = keras.utils.to_categorical(label_list, num_classes=None)

tweets_df['label'].value_counts()

#visualize word distribution
tweets_df['tw_len'] = tweets_df['text'].fillna("").apply(lambda words: len(words.split(" ")))
max_seq_len = max(tweets_df['tw_len'].values)#round(tweets_df['tw_len'].mean() + tweets_df['tw_len'].std()).astype(int)
max_seq_len


# In[5]:


#splitting the dataset into train and test in 10 ratio
x_train, x_test, y_train, y_test = train_test_split(tweet_list, label_list, test_size=0.10, random_state=50)


# In[6]:


#kfold crossvalidation using CNN functional API
tokenizer = Tokenizer(lower=True, char_level=False)
tokenizer.fit_on_texts(x_train + x_test)
word_seq_train = tokenizer.texts_to_sequences(x_train)#converting text to a vector of word indexes
word_seq_test = tokenizer.texts_to_sequences(x_test)
word_index = tokenizer.word_index

#Padding is used for different length of words in tweet,to counter this, you can #use pad_sequence() which simply pads the sequence of words with zeros

word_seq_train = sequence.pad_sequences(word_seq_train, padding='post',maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test,padding='post',maxlen=max_seq_len)


# In[7]:


#training params
num_classes = 8
batch_size = 2048
num_epochs = 100

#model parameters

hidden_dims = 32
kernel_size = 5
num_filters = 256
embed_dim = 300 
weight_decay = 0.1
#word vectors not found in the pretrained word embedding for the tweets(changed #code for error invalid arg [0.3225].. len(word_index)+1)

words_not_found = []
num_words=len(tokenizer.word_index)

embedding_matrix = np.zeros((num_words+1, embed_dim))
for word, i in word_index.items():
    #if i >= num_words:
    if i >num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


# In[8]:


#KFold crossvalidation using CNN functional API
from sklearn.metrics import classification_report 
kf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_no = 1
kfacc_per_fold=[]
kfloss_per_fold=[]
precision0 = []
precision1 = []
recall0 = []
recall1 = []
f10 = []
f11 = []
# split()  method generate indices to split data into training and test set.
for train_index, test_index in kf.split(word_seq_train, y_train):
   
   # y_train = y_train.to_numpy()
    model_input = keras.layers.Input(shape=(max_seq_len,),dtype='int32')
    x = Embedding(num_words+1, embed_dim, weights=[embedding_matrix], trainable=False)(model_input)
    x = Dropout(0.2)(x)

    conv_blocks = []
    filter_sizes = [3,4,5]
    for sz in filter_sizes:
        conv = keras.layers.Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="same",
                             activation="relu",
                             strides=1)(x)                          
                             
        conv = keras.layers.GlobalMaxPooling1D()(conv)
        conv_blocks.append(conv)
    x = keras.layers.Concatenate()(conv_blocks)
    den1 = keras.layers.Dense(250, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(den1)
    den2 = keras.layers.Dense(128, activation='relu')(x)
    model_output = keras.layers.Dense(num_classes, activation='softmax')(den2)
    ajsemodekf = keras.Model(model_input, model_output)
    optr = keras.optimizers.Adam(learning_rate=0.0003)#changed from 0.0003
    ajsemodekf.compile(loss='sparse_categorical_crossentropy', optimizer=optr, metrics=['accuracy'])
    # summarize
    print(ajsemodekf.summary())
    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'SR_Twitter_Dataset_RVCNN_Training for fold {fold_no} ...')
  
    
    histkf_fCNN= ajsemodekf.fit(np.array(word_seq_train)[train_index.astype(int)], np.array(y_train)[train_index], batch_size=2048, epochs=100, shuffle=False, verbose=1)#, class_weight=weights, use_multiprocessing = True)
    
    kfscores=ajsemodekf.evaluate(np.array(word_seq_train)[test_index.astype(int)], np.array(y_train)[test_index], batch_size=2048)
    print(f'Score for fold {fold_no}: {ajsemodekf.metrics_names[0]} of {kfscores[0]}; {ajsemodekf.metrics_names[1]} of {kfscores[1]*100}%')
    kfacc_per_fold.append(kfscores[1] * 100)
    kfloss_per_fold.append(kfscores[0])
    yhat_classes=np.argmax(ajsemodekf.predict(word_seq_test), axis=-1)
    # rounded_labels=np.argmax(y_test, axis=1)


    rep=classification_report(y_test, yhat_classes, output_dict=True)
    print('\n cLassification report:',rep)
    precision0.append(rep[str(0)]['precision'])
    precision1.append(rep[str(1)]['precision'])
    recall0.append(rep[str(0)]['recall'])
    recall1.append(rep[str(1)]['recall'])
    f10.append(rep[str(0)]['f1-score'])
    f11.append(rep[str(1)]['f1-score'])

    
    fold_no+=1


# In[19]:


print('\nSR_Twitter_data_Kfold_RVCNN_MeanTest Accuracy:', mean(kfacc_per_fold), '%')
print('\nSR_Twitter_data_Kfold_RVCNN_MeanTest Loss:', mean(kfloss_per_fold), '%')
print('\nSR_Twitter_data_Kfold_RVCNN_Standard Deviation is:', stdev(kfacc_per_fold))
print("SR_Twitter_data_Kfold_RVCNN Precision Score-0,1 -> ",np.mean(precision0),np.mean(precision1))
print("AJSE_data1_Kfold_RVCNN Recall Score-0,1 -> ",np.mean(recall0),np.mean(recall1))
print("SR_Twitter_data_Kfold_RVCNN F1 Score-0,1 -> ",np.mean(f10),np.mean(f11))

