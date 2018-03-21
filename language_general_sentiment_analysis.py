import collections
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import twython

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

#Path to the data you want to analyze
path_sample_sentiment_analysis = "data/es/sample_sentiment_analysis.json"

with open(path_sample_sentiment_analysis) as datafile:
    data = json.load(datafile)
    
for i in range (len(data)):
    data[i] = flatten(data[i])

dataframe = pd.DataFrame(data)

#Path to the csv file you want to create after the data was analyzed
path_to_csv = 'data/en.csv'
dataframe['full_text'].to_csv(path_to_csv)

full_text = pd.read_csv(path_to_csv)
full_text = list(full_text.iloc[:, 1])

sia = SentimentIntensityAnalyzer()

scores =[]

for i in range(len(full_text)):
  scores.append(sia.polarity_scores(full_text[i])['compound'])

num_bins = 9
plt.title('Sentiment Analysis About OSS by Spanish Language')
n, bins, patches = plt.hist(scores, num_bins, facecolor='blue', alpha=0.5)
plt.show()