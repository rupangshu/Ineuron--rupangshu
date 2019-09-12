# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:07:11 2019

@author: Sudhanshu2.Kumar
"""

import pandas as pd 
pd.set_option("display.max_colwidth", 200)
import numpy as np
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim
from sklearn.model_selection import train_test_split
import pickle

class IneuronPredict:

    def word_vector(self,tokens, size, model_w2v):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in tokens:
            try:
                vec += model_w2v[word].reshape((1, size))
                count += 1.
            except KeyError:  # handling the case where the token is not in vocabulary

                continue
        if count != 0:
            vec /= count
        return vec

    def getprediction(self,requestData):

        train  = pd.read_csv(r'test.csv')


        train['tidy_tweet'] = train['tweet'].str.replace("[^a-zA-Z#]", " ")

        tokenized_tweet = train['tidy_tweet'].apply(lambda x: x.split()) # tokenizing

        bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
        bow = bow_vectorizer.fit_transform(train['tidy_tweet'])
        train_bow = bow[:,:]

        model_w2v = gensim.models.Word2Vec(
                    tokenized_tweet,
                    size=200, # desired no. of features/independent variables
                    window=5, # context window size
                    min_count=2,
                    sg = 2, # 1 for skip-gram model
                    hs = 0,
                    negative = 10, # for negative sampling
                    workers= 2, # no.of cores
                    seed = 34)

        model_w2v.train(tokenized_tweet, total_examples= len(train['tidy_tweet']), epochs=20)



        xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],
                                                                  random_state=42,
                                                                  test_size=0.3)


        wordvec_arrays = np.zeros((len(tokenized_tweet), 200))

        for i in range(len(tokenized_tweet)):
            wordvec_arrays[i,:] = self.word_vector(tokenized_tweet[i], 200,model_w2v)

        wordvec_df = pd.DataFrame(wordvec_arrays)
        train_bow = bow[:,:]

        train_w2v = wordvec_df.iloc[:,:]
        test_w2v = wordvec_df.iloc[:,:]

        xtrain_w2v = train_w2v.iloc[ytrain.index,:]
        xvalid_w2v = test_w2v.iloc[yvalid.index,:]

        filename = 'finalized_model.sav'

        #testing the data

        loaded_model = pickle.load(open(filename, 'rb'))

        bow1 = pd.DataFrame([requestData])

        wordvec_arrays = np.zeros((len(bow1), 200))

        for i in range(len(bow1)):
            wordvec_arrays[i,:] = self.word_vector(bow1[i], 200,model_w2v)

        wordvec_df = pd.DataFrame(wordvec_arrays)
        test_w2v = wordvec_df.iloc[:,:]
        xvalid_w2v = test_w2v.iloc[:,:]
        prediction = loaded_model.predict(xvalid_w2v)
        return prediction