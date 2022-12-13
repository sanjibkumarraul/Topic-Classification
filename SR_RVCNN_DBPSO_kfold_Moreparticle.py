#!/usr/bin/env python
# coding: utf-8

# In[2]:


#This python program contains all the code for reading the #pretrained vectors and twitter data reading and writing the #output in command prompt
#importing required packages
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
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
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
from sklearn.model_selection import train_test_split

# For array, dataset, and visualizing
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.utils import class_weight

sns.set_style("whitegrid")
np.random.seed(0)
MAX_NB_WORDS = 180000

nltk.download('stopwords')

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
#Reading the .csv file after preprocessing
tweets_df = pd.read_csv('~/Desktop/SR_Github_link/Process_withoutnan_SRTwitter_Topic_Classification_dataset.csv',header=0,encoding='latin-1',keep_default_na=False)

    
    #Added code for adding class weight
class_weights=list(class_weight.compute_class_weight('balanced',np.unique(tweets_df['label']), tweets_df['label']))
tweets_df['label'].value_counts()
class_weights.sort()
weights={}
    #convert class_weight which is in list format to dictionary format
for index, weight in enumerate(class_weights) :
    
    weights[index]=weight


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




# In[3]:


#kfold crossvalidation using CNN functional API
x_train, x_test, y_train, y_test = train_test_split(tweet_list, label_list, test_size=0.10, random_state=50)


# Tokenizer used to convert text into a token form means for each word an index #will generate,Tokenizer utility class which can vectorize a text corpus into a #list of integers


tokenizer = Tokenizer(lower=True, char_level=False)
tokenizer.fit_on_texts(x_train + x_test)
word_seq_train = tokenizer.texts_to_sequences(x_train)#converting text to a vector of word indexes
word_seq_test = tokenizer.texts_to_sequences(x_test)
word_index = tokenizer.word_index





#Padding is used for different length of words in tweet,to counter this, you can #use pad_sequence() which simply pads the sequence of words with zeros

word_seq_train = sequence.pad_sequences(word_seq_train, padding='post',maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test,padding='post',maxlen=max_seq_len)


# In[5]:


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
#num_words = min(MAX_NB_WORDS, len(word_index))
#nb_words = min(MAX_NB_WORDS, len(word_index))

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


# In[6]:


#DBPSO code
from statistics import mean
from sklearn.model_selection import cross_validate
#from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
def call_demo(batch_size,num_epochs,num_filters,lrate,particle_rep):
    
    
    
    #kfold crossvalidation using CNN functional API
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    fold_no = 1
    
    dt_loss = []
    dt = []                 
    # split()  method generate indices to split data into training and test set.
    for train_index, test_index in kf.split(word_seq_train, y_train):

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
        den2 = keras.layers.Dense(100, activation='relu')(x)
        model_output = keras.layers.Dense(num_classes, activation='softmax')(den2)
        modelkfpso = keras.Model(model_input, model_output)
        opt = keras.optimizers.Adam(learning_rate=lrate)
        modelkfpso.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        # summarize
        print(modelkfpso.summary())

        csv_log_file = "/home/system2-user1/sanjibwork/FW_Output/CSVLoger/SR_TWNAN_CNNDBPSO_kfold"+particle_rep+".csv"
        #tensor_board_log_dir = "/home/system2-user1/sanjibwork/FW_Output/Clogs/tensorboard/FW_CNN__Fun_0.2_0.5_100ep"
        chkpt_file = "/home/system2-user1/sanjibwork/FW_Output/CheckPoint/SR_TWNAN_CNNDBPSO_kfold"+particle_rep+".h5"
        from keras.callbacks import EarlyStopping
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard

        csv_logger = CSVLogger(csv_log_file)
        #tensor_board=TensorBoard(log_dir=tensor_board_log_dir, histogram_freq=1, write_graph=True, write_images=True)
        #early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)
        check_pts = ModelCheckpoint(chkpt_file, monitor='val_accuracy', save_best_only=True, verbose=1, save_freq='epoch')

        callbacks_list = [check_pts, csv_logger]#, tensor_board]#, early_stop]

        histkfpso_fCNN= modelkfpso.fit(np.array(word_seq_train)[train_index.astype(int)], np.array(y_train)[train_index], batch_size=batch_size, validation_split=0.2, epochs=num_epochs, shuffle=False, verbose=1, callbacks = callbacks_list, class_weight=weights, use_multiprocessing = True)
        
        dt_=modelkfpso.evaluate(np.array(word_seq_train)[test_index.astype(int)], np.array(y_train)[test_index], batch_size=2048)
        
        print(f'Score for fold {fold_no}: {modelkfpso.metrics_names[0]} of {dt_[0]}; {modelkfpso.metrics_names[1]} of {dt_[1]*100}%')
        print('\nLoss value:',dt_[0])
        print('Accuracy value:',dt_[1])
        dt.append(dt_[1])       
        dt_loss.append(dt_[0])
                                    
                       
        fold_no+=1
    print("k-fold acc scores are {}".format(dt))     
    return mean(dt)                                   


# In[ ]:





# In[7]:


import random
import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, TensorBoard

 
cols = 4             #for 4 values(batc_siz,epoc,filt,lrate)
No_of_bits = 3       #how many bits for swarm size
No_of_points = 8   #swarm size 2**No_of_bits
binary_max = 8      #maximum bits 2**No_of_bits
O_Points = []

# ------------Generating Binary Numbers-----------------------
b = [ bin(i)[2:].zfill(No_of_bits) for i in range(binary_max) ]
# ------------------------------------------------------------

# -------------Setting UP B_Ponits with Binary Digits---------
B_Points = [ [b[i] for j in range(cols)] for i in range(No_of_points) ]
print(B_Points)
# ------------------------------------------------------------

for i in range(0,No_of_points):
    fltr = random.randint(32,512)
    btch_size = 2**(No_of_points-i+cols)
    #epch = random.randint(0,binary_max)
    epch = random.randint(80,150)
    #l_rate = random.uniform(0.0001, 0.001)
    l_rate = random.uniform(0.0001, 0.0006)
    O_Point_Initial = [btch_size , epch, fltr, l_rate]
    O_Points.append(O_Point_Initial)
print(O_Points)
#------------------
O_Points_current = []

#Accuracy has been taken as same number of points
P_ac_best = []
P_ac_current = []

#Number of Iteration that we want to do
maxiter = 10



fl_cnt = 0;
iter_pool_accur = []
mx_val=0                     
max_val_point = [0]*4        
for it in range(0,maxiter):
    i = 0
    if (it==0):
        P_ac_current = P_ac_best
        P_ac_best = []
        for r in O_Points:
            #Plocal_best = []
            flag = False
            batch_size = r[0]    
            num_epochs = r[1] 
            #model parameters    
            num_filters = r[2] 
            lrate = r[3]
            #weight_decay = r[4]    
             
            particle_rep = 'Sequential_'.join([str(fl_cnt)])
            fl_cnt = fl_cnt+1
            
            rt_value = call_demo(batch_size,num_epochs,num_filters,lrate, particle_rep)
            if (rt_value>mx_val):             
                mx_val = rt_value             
                max_val_point = r             
              
            
            P_ac_current.append(rt_value) 
            
            print(rt_value)
          
        P_ac_current.sort(reverse=True)
        len_a = len(O_Points)
        for ind in  range(0,len_a):
            P_ac_best.append(P_ac_current[ind])
        P_ac_current = P_ac_best
                   
        #///////////////////////// First time accuracy
        print("------------------------------------------------------------------")
        print(P_ac_best)
        print("--*********************************************---")
        
    
    else:

        cur_in = 0
        for dt in O_Points:

            Pbest = []
        #   This is the Hyper parameter which we will receive from dt by iteration
            p = [] 
            Pbest_current = [] 

            #Converting into binary using random binary encoding
            

            local_a = ''
            local_j = 0
            for d in dt:
                local_a += B_Points[cur_in][local_j]
                local_j = local_j+1
            Pbest.append(local_a)
            p.append(local_a)

            #Checking the Data appending
            ind_i = 0 
            for data in Pbest:
                ind_i = ind_i+1
                   
            ind_j = len(p[0])
            velocity = [0]*ind_j
            w = 0.7
            r1=random.randint(0,1)
            r2=random.randint(0,1)
            c1,c2= 2.0,2.0

            #Herre we finding global best among all the accruracy
            gbest = max(P_ac_best)

            upsigmoid = []
               
            for j in range (0, ind_j):
                Vmax = max(velocity)
                velocity_cal = w*velocity[j]+(c1*r1*(P_ac_best[cur_in]-int(p[0][j])))+(c2*r2*(gbest-int(p[0][j])))
                if velocity_cal>=Vmax:
                    velocity[j] = Vmax*(np.sign(velocity_cal))
                else:
                    velocity[j] = velocity_cal
                a = tf.constant(velocity_cal, dtype = tf.float32)   
                # Applying the sigmoid function and 
                # storing the result in 'b'     
                b = tf.nn.sigmoid(a, name ='sigmoid')
                with tf.compat.v1.Session() as sess: 
                     
                    sigmoid_value = sess.run(b)
                r = random.randint(0,1)
                x = 0
                if r<sigmoid_value:
                    x=1
                velocity[j] = x
            print(velocity)
            data = ''
            p = []
            for vel in velocity:
                data += str(vel)
            p.append(data)
          
            local_points = []
            p_str = p[0]
            cnt1 = 0
            # Here we  arer decoding from new updated velocity to Particles for the next iteration
            for ind in range(0, len(p_str), No_of_bits):
                data_vl = p_str[ind:ind+No_of_bits]
                
                 
                 
                for a in range(0,len(B_Points)):
                    if B_Points[a][cnt1] == data_vl:
                        
                        local_points.append(O_Points[a][cnt1])
                         
                cnt1 = cnt1+1

            print("Updated Local Points")
            print(len(local_points))
            print(local_points)
            O_Points_current.append(local_points)
            print("////////////////////////////////////////////")
            
            
            particle_rep = 'Sequential_'.join([str(fl_cnt)])
            fl_cnt = fl_cnt+1
            rt_value = call_demo(local_points[0],local_points[1],local_points[2],local_points[3],particle_rep)
            if(rt_value>P_ac_best[cur_in]):
                P_ac_best[cur_in] = rt_value
                mx_val = rt_value                          
                max_val_point = local_points               
             
            gbest = max(P_ac_best)
            cur_in = cur_in+1
            if len(O_Points) == len(O_Points_current):
                O_Points = O_Points_current
                print(" O_Points Size after Each Iteration")
                print(len(O_Points))
                O_Points_current = []
            iter_pool_accur.append(gbest)
            i = i+1 
        print("-------------------------------------------------------")
        print(P_ac_best)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print()
print("------------&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&----------------")
best_acc = max(iter_pool_accur)
ind2=0
for dt in iter_pool_accur:
    if best_acc==dt:
        break;
    ind2 = ind2+1
    ind2 = ind2%len(O_Points)  
print("Particle Number:")

print(max_val_point)
print(best_acc)

