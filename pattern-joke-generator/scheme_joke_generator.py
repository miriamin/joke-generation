
import mysql.connector
import nltk
from nltk.corpus import wordnet as wn
from nltk.probability import FreqDist
import pandas as pd
import numpy as np
import time
import datetime
import math
from sklearn import preprocessing


news = 'eng_news_2016_1m'


def get_candidates(x):
    start = time.time()
    print("Getting candidates …")
    candidates = []
    # find all Z's (i.e. attributes) connected to X
    zs = pos_filter(get_left_cooccurrences(x), 'a')
    for z in zs:
        ys = pos_filter(get_right_cooccurrences(z), 'n')
        for y in ys:
            candidate = (x, y, z)
            candidates.append(candidate)
    end = time.time()
    total = end - start
    print("Found " + str(len(candidates)) + " joke candidates in " + str(total) +" sec")        
    return candidates 



def relatedness(word1,word2):
    try:# get absolute cooccurrence frequency of word 1 and word 2
        abs_cooc_freq = get_cooc_freq(word1,word2)
    except: # return 0 if no cooccurrence is recorded
        abs_cooc_freq = 0
    # get number of all cooccurrences in database    
    db = mysql.connector.connect(host='localhost', database=news, user='root', password='cts')
    cursor = db.cursor()
    cursor.execute("SELECT SUM(freq) as sum FROM co_n",)
    all_cooc_freq = int(cursor.fetchall()[0][0])
    # calculate relative cooccurrence frequeny as ratio of absolute cooc frequency and number of all coocs
    rel_cooc_freq = abs_cooc_freq/all_cooc_freq
    return rel_cooc_freq




def surprisal(word):
    db = mysql.connector.connect(host='localhost', database=news, user='root', password='cts')
    cursor = db.cursor()
    # get word
    cursor.execute("SELECT freq FROM words WHERE word = %s", (word,))
    try: # get word frequency and calculate reciproce
        abs_freq = cursor.fetchall()[0][0]
        surprisal = 1/abs_freq
    except: # set suprisal 0 if word not found in corpus
        surprisal = 0    
    return surprisal



def ambiguity(word):
    try:
        senses = len(wn.synsets(word))
        ambiguity = 1/senses
    except: ambiguity = 0    
    return ambiguity



def dissimilarity(word1,word2):
    try:
        dissimilarity = 1/similarity(word1,word2)
    except:
        dissimilarity = 0
    return dissimilarity



def similarity(word1,word2):
    word_vectors = get_word_vectors(word1,word2)
    # calculate cosinus similary as dot product of the two vectors
    # source: https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
    similarity = word_vectors.loc[word1].dot(word_vectors.loc[word2])
    return similarity


def get_word_id(word):
    try: 
        db = mysql.connector.connect(host='localhost', database=news,user='root',password='cts')
        cursor = db.cursor()
        cursor.execute("SELECT w_id FROM words WHERE word = %s", (word,))
        word_id = cursor.fetchall()[0][0]
    except: 
        print("Word not found")
        pass
    return word_id



def get_word_from_id(word_id):
    db = mysql.connector.connect(host='localhost', database=news,user='root',password='cts')
    cursor = db.cursor()
    cursor.execute("SELECT word FROM words WHERE w_id = %s", (word_id,))
    word = cursor.fetchall()[0][0]
    return word



def get_left_cooccurrences(word):
    db = mysql.connector.connect(host='localhost', database=news,user='root',password='cts')
    cursor = db.cursor()
    word_id = get_word_id(word)
    cursor.execute("SELECT w1_id FROM co_n WHERE w2_id = %s" , (word_id,))
    result_list = cursor.fetchall()
    cooc_list = [get_word_from_id(result[0]) for result in result_list]
    return cooc_list



def get_right_cooccurrences(word):
    db = mysql.connector.connect(host='localhost', database=news,user='root',password='cts')
    cursor = db.cursor()
    word_id = get_word_id(word)
    cursor.execute("SELECT w2_id FROM co_n WHERE w1_id = %s" , (word_id,))
    result_list = cursor.fetchall()
    cooc_list = [get_word_from_id(result[0]) for result in result_list]
    return cooc_list


def pos_filter(word_list, pos_tag): #can be either 'a', 'n', 's', 'r' or 'v'
    # https://wordnet.princeton.edu/documentation/wndb5wn
    filtered_word_list = []
    for word in word_list:
        pos_list = [synset.pos() for synset in wn.synsets(word)] # get POS-tags for all senses of a word
        if pos_tag in pos_list: # if one sense of a word is of the relevant POS-tag, the word passes the filter
            filtered_word_list.append(word)
    return filtered_word_list   


def get_cooc_freq(word1,word2):
    db = mysql.connector.connect(host='localhost', database=news,user='root',password='cts')
    cursor = db.cursor()
    w1_id = get_word_id(word1)
    w2_id = get_word_id(word2)
    try: 
        cursor.execute("SELECT freq FROM co_n WHERE w1_id = %s AND w2_id = %s" , (w1_id, w2_id))
        cooc_freq = cursor.fetchall()[0][0]
    except:
        cooc_freq = 0
    return cooc_freq
    


def get_word_vectors(word1,word2):
    #get all words that are left neighbors of one of the both words in the corpus
    attribute_candidates = set(get_left_cooccurrences(word1) + get_left_cooccurrences(word2))
    # filter list for adjectives
    attribute_list = pos_filter(attribute_candidates, 'a')
    # load DataFrame with word1 + word2 as index and all attributes as column names
    df = pd.DataFrame([word1,word2], columns=['type'])
    df2=pd.DataFrame(columns=attribute_list)
    vectors=pd.concat([df,df2])
    vectors = vectors.set_index('type')
    # calcuate and normaliaze vectors for each word
    for word in [word1,word2]:
        word_vector = [get_cooc_freq(attribute,word) for attribute in vectors.columns]
        vectors.loc[word] = word_vector 
        # normalize over term frequency with euclidian norm/L2 norm: 
        # calculate square root of sum of squares of freqs
        # divide all values by this
        # https://towardsdatascience.com/overview-of-text-similarity-metrics-3397c4601f50
        vectors.loc[word] = vectors.loc[word] **2 
        l2_norm = math.sqrt(vectors.loc[word].sum(axis=0))
        vectors.loc[word] = vectors.loc[word]/l2_norm   
    return vectors


def get_candidate_scores(candidates,seed):
     # set up DataFrame
    candidates_df = pd.DataFrame(candidates, columns=['X','Y','Z'])
    candidates_df['relatedness(Y,Z)'] = np.nan
    candidates_df['relatedness(X,Z)'] = np.nan
    candidates_df['surprisal(Z)'] = np.nan
    candidates_df['ambiguity(Z)'] = np.nan
    candidates_df['dissimilarity(x,y)'] = np.nan
    start = time.time()
    # Calcualte scores for all candidates
    for i in range(len(candidates)):
        try: 
            print("Calculating candidate " + str(i) +"/"+ str(len(candidates)))
            x = candidates[i][0]
            y = candidates[i][1]
            z = candidates[i][2]
            candidates_df.at[i,'relatedness(Y,Z)'] = relatedness(z,y)
            candidates_df.at[i,'relatedness(X,Z)'] = relatedness(z,x)
            candidates_df.at[i,'surprisal(Z)']= surprisal(x)
            candidates_df.at[i,'ambiguity(Z)'] = ambiguity(z)
            try: 
                candidates_df.at[i,'dissimilarity(x,y)'] = dissimilarity(x,y)
            except:     
                candidates_df.at[i,'dissimilarity(x,y)'] = 0
        except: 
            pass
        if i % 100 == 0:
            intermediate = time.time()
            print("Running since " + str((intermediate-start)/60) + " min")
            candidates_df.to_csv('output/'+ seed + ".csv")
    normalized_values=preprocessing.normalize((candidates_df[candidates_df.columns[3:9]]))
    df_normalized = candidates_df
    df_normalized[df_normalized.columns[3:9]]=normalized_values
    df_normalized['Score']=df_normalized['relatedness(Y,Z)']*df_normalized['relatedness(X,Z)']*df_normalized['surprisal(Z)']*df_normalized['dissimilarity(x,y)']
    df_normalized = df_normalized.sort_values(by=['Score'], ascending=False)  
    candidates_df = candidates_df.sort_values(by=['Score'], ascending=False)    
    candidates_df.to_csv('output/'+ seed + ".csv")
    return candidates_df
        


# seedword is the X variable, top_k is the number of jokes that will be outputtet finally
# returns list with top k jokes, prints them and saves them in a file
def generate_jokes(seedword,top_k):
    outfile = 'output/'+ seedword + '_jokes'+'.txt'
    start = time.time()
    try: # check if file already exists
        filename = 'output/'+ seedword + '.csv'
        jokes_df = pd.read_csv(filename)
    except: # find candidate triples and calculate scores
        candidates = get_candidates(seedword)
        jokes_df = get_candidate_scores(candidates,seedword)
        print("Error")
    top_jokes = []
    num_jokes = len(jokes_df.index)
    if num_jokes > top_k:
        num_output_jokes = top_k
    else: 
        num_output_jokes = num_jokes
    for i in range(num_jokes):
        x = jokes_df.at[i,'X']
        y = jokes_df.at[i,'Y']
        z = jokes_df.at[i,'Z']
        joke = "I like my " + x + " like I like my " + y + ", " + z + "."
        top_jokes.append(joke)
        #print(joke)
    with open(outfile, 'w') as outfile:
        for joke in top_jokes:
            outfile.write(joke + '\n')
    end = time.time()
    total = end-start
    print(str(total/60) + " min")
    return top_jokes


