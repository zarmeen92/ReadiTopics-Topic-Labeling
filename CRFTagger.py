#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pickle
from sklearn_crfsuite import CRF


# In[5]:


def return_suffix(word):
    suffix=''
    wlen = len(word)
    if wlen <=1:
        suffix = word
    elif wlen > 1 and wlen <=2:
        suffix = word[wlen-1]
        
    elif wlen == 3:
        suffix = word[wlen-2:wlen]
    else:
        suffix = word[wlen-3:wlen]
    return suffix

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'length':len(sentence[index]),
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'suffix': return_suffix(sentence[index]),
        'prev_word': '' if index == 0 else sentence[index - 1],
        'prev_word1': '' if index < 2 else sentence[index - 2],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1]
           }


# In[8]:


filename = 'C:/Users/Zarmeen/Documents/Phd work/pos_models/crf_1000000_model_25Mar.sav'
model = pickle.load(open(filename, 'rb'))
def pos_tag(s):
    sentence = s.split()
    #print(sentence)
    sentence_features = [features(sentence, index) for index in range(len(sentence))]
    return list(zip(sentence, model.predict([sentence_features])[0]))
 


# In[ ]:




