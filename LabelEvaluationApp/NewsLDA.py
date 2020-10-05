#!/usr/bin/env python
# coding: utf-8

# In[19]:


from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import gensim.models as g
import urduhack
from urduhack import stop_words
from urduhack import tokenization as tok
from urduhack import preprocess
from urduhack import utils
from urduhack import normalization as norm
from utilities import words
import re
from gensim.models import Word2Vec,KeyedVectors
import CRFTagger
import numpy as np
import pandas as pd
import spacy
from sklearn import metrics
import collections
from scipy.cluster import  hierarchy
from matplotlib import pyplot as plt
from kneed import KneeLocator
import time
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel


# In[20]:


stopwords = list(stop_words.STOP_WORDS)
phraser_model = g.Phrases.load('fourgram-urdunews-mod.sav')

# In[31]:
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
def tweet_preprocessing(tweet):
    tweet = tweet.replace('\n'," ")
    tweet = tweet.replace('\r'," ")
    tweet = preprocess.normalize_whitespace(tweet)
    tweet = preprocess.remove_punctuation(tweet)
    tweet = norm.remove_diacritics(tweet)
    tweet = emoji_pattern.sub(r'', tweet)
    tweet = words.fix_join_words(tweet)
    tweet_clean = re.sub(r"http\S+", "", tweet)
    tweet_clean = re.sub('@[^\s]+','',tweet_clean)
    nonalpha = re.compile(r"[a-zA-Z0-9.@#_:)(-]")
    cleanText = re.sub(nonalpha,'',tweet_clean).strip()#remove english alphabets
    return cleanText
def split_into_words(text):
    
    return str(text).split()
def get_words_from_tags(postags):
    words = []
    tags = []
    for u,v in postags:
        words.append(u)
        tags.append(v)
    return words,tags
#function taken from Textacy library to identify noun, verb and prepositional phrases
def pos_regex_matches(doc, pattern,tgs):
    """
    Extract sequences of consecutive tokens from a spacy-parsed doc whose
    part-of-speech tags match the specified regex pattern.

    Args:
        doc (``textacy.Doc`` or ``spacy.Doc`` or ``spacy.Span``)
        pattern (str): Pattern of consecutive POS tags whose corresponding words
            are to be extracted, inspired by the regex patterns used in NLTK's
            `nltk.chunk.regexp`. Tags are uppercase, from the universal tag set;
            delimited by < and >, which are basically converted to parentheses
            with spaces as needed to correctly extract matching word sequences;
            white space in the input doesn't matter.

            Examples (see ``constants.POS_REGEX_PATTERNS``):

            * noun phrase: r'<DET>? (<NOUN>+ <ADP|CONJ>)* <NOUN>+'
            * compound nouns: r'<NOUN>+'
            * verb phrase: r'<VERB>?<ADV>*<VERB>+'
            * prepositional phrase: r'<PREP> <DET>? (<NOUN>+<ADP>)* <NOUN>+'

    Yields:
        ``spacy.Span``: the next span of consecutive tokens from ``doc`` whose
        parts-of-speech match ``pattern``, in order of apperance
    """
    # standardize and transform the regular expression pattern...
    pattern = re.sub(r'\s', '', pattern)
    pattern = re.sub(r'<([A-Z]+)\|([A-Z]+)>', r'( (\1|\2))', pattern)
    pattern = re.sub(r'<([A-Z]+)>', r'( \1)', pattern)

    tags = ' ' + ' '.join(tgs)

    for m in re.finditer(pattern, tags):
        yield doc[tags[0:m.start()].count(' '):tags[0:m.end()].count(' ')]
def form_phrase(x):
    return "_".join(x)
def return_noun_phrase(text):
    tags = CRFTagger.pos_tag(text)
    noun_phrases = []
    #print(tags)
    words,tgs = get_words_from_tags(tags)
    patern = r'<NN|PN>+'
    lists = pos_regex_matches(words, patern,tgs)
    for x in lists:
        noun_phrases.append(form_phrase(x))
    return noun_phrases
def filter_noun_adj_from_phrase(phrases):
    noun_phrases = []
    adj_phrases=[]
    for x in phrases:
        words = x.split("_")
        tags = CRFTagger.pos_tag(" ".join(words))
        words,tgs = get_words_from_tags(tags)
        patern = r'<A|SC>*<ADJ>*<NN|PN>+'
        lists = pos_regex_matches(words, patern,tgs)
        for j in lists:
            print(j)
            noun_phrases.append(form_phrase(j))
   
    return noun_phrases


def return_nouns_sent(nn_list):
    return " ".join(nn_list)
def return_phrase_sent(nn_list):
    return " ".join(nn_list)
def return_gensim_phrase(sentence):
    phrase = phraser_model[sentence.split()]
    return phrase

def list_phrases_cluster(cluster_idx,df,key):
    df_sub = df[df[key] == cluster_idx]
    hashs = []
    for index,row in df_sub.iterrows():
        #print(row.Hashtags)
        for x in row.Noun_Phrases:
            hashs.append(x)
    
    return collections.Counter(hashs)
def find_common_strings(mylist):
    #retain unique words only
    unique_list = []
    for i in mylist:
        myset = set(i)
        unique_list.append(" ".join(myset))
    print(unique_list)
    words = []
    for x in unique_list:
        for j in x.split():
            words.append(j)
    cnt = collections.Counter(words)
    common_words = []
    for u,v in cnt.items():
        if(v >= len(mylist)):
            common_words.append(u)
    return common_words
def return_clusters_title(cluster_idx,dataset,col):
    hash_cluster1 = list_phrases_cluster(cluster_idx,dataset,col) 
    top10  = hash_cluster1.most_common(3)
    dataset_sub = dataset[dataset[col] == cluster_idx]
    cnt_vectorizer = TfidfVectorizer(ngram_range=(3,15))
    vec = cnt_vectorizer.fit_transform(dataset_sub['Title'])
    scores = (vec.toarray()) 
    # Getting top ranking features 
    sums = vec.sum(axis = 0) 
    data1 = [] 
    features = cnt_vectorizer.get_feature_names()
    for col, term in enumerate(features): 
        data1.append((term, sums[0, col] )) 
    ranking = pd.DataFrame(data1, columns = ['term', 'rank'])
    ranking['wCount'] = ranking['term'].apply(lambda x: len(x.split()))
    ranking = ranking.sort_values(['rank','wCount'],ascending=[False,True])
    print(ranking.head(5))
    my_top_4=[]
    for u,v in top10:
        my_top_4.append(u)
    my_top_4 = " ".join(my_top_4)
    my_top_4 = my_top_4.replace("_"," ")

    #print(my_top_4)
    test_list = (my_top_4.split())
    df = pd.DataFrame(columns=ranking.columns) #empty dataframe to add filtered rows
    for index,row in ranking.iterrows():
        # using list comprehension 
        # checking if string contains list element 
        res = all(ele in row['term'] for ele in test_list) 

        # print result 
        if res == True:
            df.loc[len(df)]=[row['term'],row['rank'],row['wCount']] 
    df = df.sort_values(['rank','wCount'],ascending=[False,True])
    #df = df.sort_values(['wCount'])
    if len(df)>0:
        return df.iloc[0]['term']
    else:
        return "No title constructed"


def apply_kmeans(text):
    text = text.strip().split('\n')
    print(len(text))
    dataset = pd.DataFrame({'Title':text})
    #form phrases
    phrases = []
    for doc in dataset.Title:
        phrases.append(return_gensim_phrase(tweet_preprocessing(doc)))
    dataset['Phrases'] =phrases
    dataset['Phrases_Sent'] =dataset['Phrases'].apply(return_phrase_sent)
    dataset['Noun_Phrases'] = dataset['Phrases'].apply(filter_noun_adj_from_phrase)
    dataset['noun_sent'] = dataset['Noun_Phrases'].apply(return_nouns_sent)
    cnt_vectorizer = TfidfVectorizer(min_df=0.03)
    vec = cnt_vectorizer.fit_transform(dataset['noun_sent'])
    #elbow curve for optimal clusters
    #Elbow curve method
    X = vec.todense()
    Sum_of_squared_distances = []
    K = range(1,len(text))
    for k in K:
        print('K is %s'%k)
        km = KMeans(n_clusters=k,random_state=3)
        km = km.fit(X)
        print(km.inertia_)
        Sum_of_squared_distances.append(km.inertia_)
    kn = KneeLocator(K, Sum_of_squared_distances, curve='convex', direction='decreasing')
    print(kn.knee)
    
    if(kn.knee == None):
        nclusters = 2
    else:
        nclusters = kn.knee
    km = KMeans(nclusters,random_state=3)
    km.fit(X)
    labels = km.labels_
    dataset['km_label'] = labels
    lbl=[]
    mylist=[]
    for i in range(0,dataset['km_label'].max()+1):
        title = return_clusters_title(i,dataset,'km_label')
        print(title)
        lbl.append(i)
        mylist.append(title)
    cluster_lbl = pd.DataFrame({'Cluster':lbl,'Label':mylist})

    return dataset,cluster_lbl


