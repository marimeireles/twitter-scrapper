import collections
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import twython
import pandas as pd

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

with open('data/total_data.py') as datafile:
    data = json.load(datafile)
    
for i in range(len(data)):
    data[i] = flatten(data[i])

dataframe = pd.DataFrame(data)

time_zone_full_text = dataframe.loc[:, ['user_time_zone', 'full_text']]
time_zone_full_text = time_zone_full_text.mask(time_zone_full_text.eq('None')).dropna()

time_zone_full_text = time_zone_full_text.groupby('user_time_zone')
time_zone_counter = time_zone_full_text.count().sort_values(by='full_text', ascending=False)
time_zone_counter = time_zone_counter[:11]

#print(time_zone_full_text.groups)
#print(time_zone_counter.index)

sia = SentimentIntensityAnalyzer()

dict_sorted_time_zone_full_text = {}

for i in time_zone_counter.index:
    sorted_time_zone_full_text = time_zone_full_text.get_group(i)

    scores = []

    for j in sorted_time_zone_full_text.values:
        #print(j)
        scores.append(sia.polarity_scores(j[1])['compound'])

    dict_sorted_time_zone_full_text[i] = scores
    #print(dict_sorted_time_zone_full_text[i])

plt.boxplot(dict_sorted_time_zone_full_text.values(), labels=dict_sorted_time_zone_full_text.keys(), color='blue')
plt.xticks(range(1, 12), dict_sorted_time_zone_full_text.keys(), rotation=45, ha='right')
plt.show()
 