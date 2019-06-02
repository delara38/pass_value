#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge,  BayesianRidge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import requests
import math
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


#open up matches info and make a list of all the match ids
tournament = requests.get('https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/43.json').json()

match_ids = []
for g in tournament:
    m_id = g['match_id']
    match_ids.append(m_id)


# In[3]:


path = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/{}.json'

n = 0
X = []
for ids in match_ids:

    events = requests.get(path.format(ids)).json()
    
    passes =  [x for x in events if x['type']['name'] == "Pass" and x['position']['name'] != 'Goalkeeper']
    n += len(passes)
    for p in range(len(events)):
        if events[p]['type']['name'] == 'Pass' and events[p]['position']['name'] != 'Goalkeeper':
            minute = events[p]['minute']
            second = events[p]['second']
            x = events[p]['pass']['end_location'][0]
            y = events[p]['pass']['end_location'][1]
            pos = events[p]['possession_team']['id']
            angle = events[p]['pass']['angle']
            distance = events[p]['pass']['length']
            try:
                speed = distance/events[p]['duration']
            except:
                speed = distance
            goal = 0
            ga = 0
            for i in range(p + 1, len(events)):
                if (events[i]['minute'] < (minute+1)) or (events[i]['minute'] == (minute+1) and events[i]['second'] < second):
                    if events[i]['type']['name'] == 'Shot':
                            if events[i]['shot']['outcome']['name'] == 'Goal':
                                if events[i]['possession_team']['id'] == pos:
                                    goal += 1
                                else:
                                    ga += 1
                else:
                    break
            gd = goal - ga
            if gd > 1:
                gd = 1
            elif gd < -1:
                gd = -1
            
            X.append([ids,x,y,angle, distance, speed, gd])
        else:
            pass
len(X)


# In[ ]:


df = pd.DataFrame(X)


# In[ ]:
print('process 1 starting')

all_players = {}
c = []
h = 1
for ids in match_ids:
    train = df[df[0] != ids]
    predict = df[df[0] == ids]
    train_x = train.iloc[:,1:6].values.reshape(-1,5)
    train_y = train.iloc[:,6].values.reshape(-1,1)
    print("training model {}".format(h))
    model =  MLPClassifier(activation='tanh',hidden_layer_sizes=(5,5,5,1))
    model.fit(train_x, train_y)
    print("model {} trained".format(h))
    h = h+ 1
    events = requests.get(path.format(ids)).json()
    for p in range(len(events)):
        if events[p]['type']['name'] == 'Pass' and events[p]['position']['name'] != 'Goalkeeper':
         
            player_id = events[p]['player']['id']
            player_name = events[p]['player']['name']
            
            minute = events[p]['minute']
            second = events[p]['second']
            x = events[p]['pass']['end_location'][0]
            y = events[p]['pass']['end_location'][1]
            pos = events[p]['possession_team']['id']
            angle = events[p]['pass']['angle']
            distance = events[p]['pass']['length']
            try:
                speed = distance/events[p]['duration']
            except:
                speed = distance
            
            inputs = np.array([x,y,angle, distance, speed]).reshape(1,5)
            chance = model.predict_proba(inputs)[0][0]
            if chance > 1:
                chance = 1
            

            c.append(chance)
        
            if player_id in all_players:
                all_players[player_id]['PV'] += chance
                all_players[player_id]['Passes'] += 1
                
                if ids in all_players[player_id]['games']:
                    pass
                else:
                    all_players[player_id]['games'].append(ids)
            else:
                all_players[player_id] = {}
                all_players[player_id]['PV'] = chance
                all_players[player_id]['Passes']  = 1
                all_players[player_id]['Name'] = player_name  
                all_players[player_id]['Team'] = events[p]['possession_team']['name']
                all_players[player_id]['Position'] = events[p]['position']['name'].split(" ")[-1]
                all_players[player_id]['games'] = []
                all_players[player_id]['games'].append(ids)


# In[ ]:


for players in all_players:
    all_players[players]['games'] = len(all_players[players]['games'])


# In[ ]:


#sns.distplot(c)


# In[ ]:


all_players = pd.DataFrame(all_players).transpose().sort_values(by='PV', ascending=False)
all_players['PV/pass'] = all_players['PV'] / all_players['Passes'] 
print(all_players.sort_values(by='PV', ascending = False).head())


# In[ ]:


average = all_players['PV/pass'].mean()

all_players = all_players.transpose().to_dict()


# In[ ]:


to_add = 100
for player in all_players:
    new_val = (all_players[player]['PV'] + average * to_add) / (all_players[player]['Passes'] + to_add)
    all_players[player]['rPV/pass'] = round(new_val * 100,2)
all_players = pd.DataFrame(all_players).transpose()


# In[ ]:


all_players['rPV/pass'] = all_players['rPV/pass'].astype(float)
all_players.PV = all_players.PV.astype('float')
all_players['Passes'] = all_players['Passes'].astype(float)
all_players['PV/pass'] = all_players['PV/pass'].astype(float)


# In[ ]:


print(all_players.sort_values(by='rPV/pass',ascending=False).head())


# In[ ]:


all_players['Position'].unique()


# In[ ]:


defense = all_players["Back" == all_players['Position']]
forwards = all_players[("Forward" == all_players['Position']) | ("Striker" == all_players['Position']) | ("Wing" == all_players['Position'])]
mid = all_players["Midfield" == all_players['Position']]
print(len(mid) + len(forwards) + len(defense))


# In[ ]:


len(all_players)


# In[ ]:


defense['Rating'] = (defense['rPV/pass'] - defense['rPV/pass'].mean()) / defense['rPV/pass'].std()
forwards['Rating'] = ((forwards['rPV/pass'] - forwards['rPV/pass'].mean()) / forwards['rPV/pass'].std())
mid['Rating'] = (mid['rPV/pass'] - mid['rPV/pass'].mean()) / mid['rPV/pass'].std()


# In[ ]:


all_players = defense.append(forwards)
all_players = all_players.append(mid)
all_players = all_players.sort_values(by=['Rating'], ascending=False)
all_players = all_players.reset_index()
all_players.pop('index')
all_players.head()


all_players.to_csv('player_stats.csv')
# In[ ]:


#sns.distplot(all_players['rPV/pass'])


# In[ ]:




