# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from os import getcwd, chdir
getcwd()
chdir("C:/Users/jooyon/customer_segmentation")

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import scipy
import scipy.sparse
import matplotlib.pyplot as plt
import nltk
import random
from wordcloud import WordCloud

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

df = pd.read_csv("./data/data.csv", encoding="ISO-8859-1")
df.info()
df.head()
df.columns

df[df['InvoiceNo'].str.startswith("C")]

# handle cancellation as new feature
df['Cancelled']=df['InvoiceNo'].str.startswith("C")
df['Cancelled'] = df['Cancelled'].fillna(False)
df.head()

len(list(df['Description'].unique()))

# handle incorrect description
df = df[df['Description'].str.startswith('?')==False]
df = df[df['Description'].str.isupper()==True]
df = df[df['Description'].str.contains('LOST') == False]
df = df[df['CustomerID'].notnull()]
df['CustomerID'] = df['CustomerID'].astype(int)

df.head()
df.info()

# Convert Invoice Number to integer
df['InvoiceNo'].replace(to_replace='\D+', value=r'', regex=True, inplace=True)
df['InvoiceNo'] = df['InvoiceNo'].astype('int')

# remove shipping invoices
df=df[(df['StockCode'] != 'DOT') & (df['StockCode'] != 'POST')]
df.drop('StockCode', inplace=True, axis=1)

df.info()

# remove outliers by qty

























