import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

def calc_humor_at_k(df):
    k = len(df.index)
    humor_at_k = df["Rating"].sum()/k
    print("humorousness@"+str(k)+" =")
    return humor_at_k
    

def return_top_k_jokes(seed,k):
    file = '../pattern-joke-generator/output/' + seed + '_jokes.txt'
    with open(file, 'r') as file:
        data = file.read()
    top_k = data.split('\n')[:k]
    df = pd.DataFrame(top_k,columns=['Jokes'])
    pd.set_option('display.max_colwidth', -1)
    return df
    
def assign_evaluation(joke_df,evaluation_list):
    joke_df['Rating']=evaluation_list
    return joke_df    

def load_joke_df(scenario):
	path = "../neural-joke-generator/output/"
	df = pd.read_csv(path+scenario+'.csv', delimiter=';')
	df['ID'] = df['ID'].apply(lambda x: scenario + '-' +  str(x))
	df.set_index('ID', drop=True)
	pd.set_option('display.max_colwidth', -1)
	return df
