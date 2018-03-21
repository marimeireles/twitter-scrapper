import collections
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    
for i in range (len(data)):
    data[i] = flatten(data[i])

dataframe = pd.DataFrame(data)

user_time_zone = dataframe['user_time_zone']
print(user_time_zone)
number_user_time_zone = Counter(user_time_zone)
del number_user_time_zone[None]
number_user_time_zone = number_user_time_zone.most_common(11)
 
Time_Zone = list(zip(*number_user_time_zone))[0]
Population = list(zip(*number_user_time_zone))[1]
x_pos = np.arange(len(Time_Zone))

plt.title('People Talking About OSS by Timezone')
plt.bar(x_pos, Population, align='center')
plt.xticks(x_pos, Time_Zone, rotation=45, ha='right')
plt.show()

print(number_user_time_zone)