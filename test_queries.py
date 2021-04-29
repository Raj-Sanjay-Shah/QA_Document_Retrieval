
# Raj Sanjay Shah
#
# Please change the file paths as required
address_of_dictionary="dict.txt"
address_of_docnumber_title_mapping="dict_name.txt"
address_of_vocabulary="vocab.txt"
document_text="document.txt"
file_to_store_questions = "toughness.txt"
questions_file = 'qanta.train.json'
# File to save the top three documents for the "Predict the human difficulty of a question"
top_3_docs = 'app.json'
# Number of Questions to test
total = 10
# Starting point for the questions in the set of questions
start = 0

import sys
import enchant
import re,nltk,collections
from nltk.util import ngrams
from bs4 import BeautifulSoup as BSoup
from matplotlib import pyplot as plt
from spellchecker import SpellChecker
import numpy as np
import pandas as pd
import math
import string
import pickle
from nltk.corpus import wordnet

new_dict={}
f = open(address_of_dictionary,"rb")
with f as filehandle:
    new_dict = pickle.load(filehandle)

new_dict_document={}
f = open(document_text,"rb")
with f as filehandle:
    new_dict_document = pickle.load(filehandle)

new_dict_name={}
f = open(address_of_docnumber_title_mapping,"rb")
with f as filehandle:
    new_dict_name = pickle.load(filehandle)

vocab=[]
f = open(address_of_vocabulary,"rb")
with f as filehandle:
    vocab = pickle.load(filehandle)
spell = SpellChecker(language=None,distance=1,case_sensitive=False)
spell.word_frequency.load_words(vocab)



# Creates an inverted index for the query
def add_to_inverted_index(text,dic,docnumber):
# Removing punctuation and converting everything to lower case
    tokens=nltk.word_tokenize(text)
    tok=[x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]
    token = [w.lower() for w in tok if w.isalpha()]
# Creating the inverted index with frequencies
    for a in token:
        if(a in dic):
            ans=dic[a].pop()
            if(docnumber != ans[0] ):
                dic[a].append(ans)
                dic[a].append([docnumber,1])
            else:
                ans[1]=ans[1]+1
                dic[a].append(ans)
        else:
            dic[a]=[[docnumber,1]]



# Calculates the score using the lnc.ltc scoring scheme,
def score_calc(weight_query,weight_doc,k):
    score=0
    temp1=0
    test=0
    # print(len(weight_doc))
    for i in weight_doc:
        if (i==0):
            test=test+1
    # print(test)
# Return zero if the number of words in query but not in document are greater than threshhold value
    if(test>len(weight_query)*k):
        return 0

    for i in weight_query:
        temp1=temp1+i**2

    norm1=math.sqrt(temp1)
    temp2=0

    for i in weight_doc:
        temp2=temp2+i**2

    if (temp2==0):
        return 0

    norm2=math.sqrt(temp2)

    for i in range(len(weight_query)):
        score=score+weight_query[i]*weight_doc[i]/(norm1*norm2)
    return score



# Function to specify key for sorting
def sortSecond(val):
    return val[1]
def sortThird(val):
    return val[2]


# Function that calculates the weights for each document and query
# and then calls score_calc to calculate the document score
def query_docs(dic,query,No_of_docs,k):
# dic contains the inverted index
# No_of_docs contains the total number of documents for easy calculation
# (k* number of query words) is the minimum number of words that each document should contain
    myQuery={}
    add_to_inverted_index(query,myQuery,0)
    weight=[]
    doc = [[0 for i in range(len(myQuery))] for j in range(No_of_docs)]
    score=[[-1,-1]]*No_of_docs
    i=0
    for a in myQuery:
        if a in dic:
            tf_query=math.log2(myQuery[a][0][1])+1
            df_query=len(dic[a])
            idf_query=math.log2(No_of_docs/df_query)
            weight.append(tf_query*idf_query)
            for x in dic[a]:
                k=x[0]
                doc[k][i]=math.log2(x[1]+1)
        else:
            weight.append(0.0)
        i=i+1

    for i in range(len(doc)):
        score[i]=[i,score_calc(weight,doc[i],k)]

    score.sort(key = sortSecond, reverse = True)

    for i in range(len(score)):
        if (score[i][1]==0.0):
            del score[i:]
            break
    return score





# query_check calls query_docs and checks if we have atleast ten results or not, if not, then decreases value of k
def query_check(query,new_dict,k,new_dict_name):
    tokens=nltk.word_tokenize(query)
    x=query_docs(new_dict,query,len(new_dict_name),k)
    g=[]
# Heuristics to deal with cases when the number of results returned are less than ten
# If the number of cases are less than ten, we decrease the values of k by a factor of two.
    while (len(x)<10 and k>0.01):
        if(len(tokens)>1):
            k=float(k)/2
            x=query_docs(new_dict,query,len(new_dict_name),k)
        else:
            break
    x.sort(key = sortSecond, reverse = True)
    for i in x:
        g.append([new_dict_name[i[0]],i[1]])
    return g





# qury provides modularity to the code,the user does not worry about the value of k
def qury(query,new_dict,new_dict_name):
    return query_check(query,new_dict,0.5,new_dict_name)

#
# Including functionality of finding equivalent classes if the number of documents are too less
def query_docs_synonyms(dic,query,No_of_docs,k):
    myQuery={}
    add_to_inverted_index(query,myQuery,0)

    weight=[]
    s=0
    for a in myQuery:
        synonms=[]
        s=s+1
        for syn in wordnet.synsets(a):
            for l in syn.lemmas():
                s=s+1
    doc = [[0 for i in range(s)] for j in range(No_of_docs)]
    score=[[-1,-1]]*No_of_docs
    i=0
    for a in myQuery:
        synonyms = []
        synonyms.append(a)
        for syn in wordnet.synsets(a):

            for l in syn.lemmas():
                synonyms.append(l.name())
        for wo in synonyms:
            if wo in dic:
                tf_query=math.log2(myQuery[a][0][1])+1
                df_query=len(dic[a])
                idf_query=math.log2(No_of_docs/df_query)
                weight.append(tf_query*idf_query)
                for x in dic[wo]:
                    k=x[0]
                    doc[k][i]=math.log2(x[1]+1)

            else:
                weight.append(0.0)
            i=i+1
    for i in range(len(doc)):
        score[i]=[i,score_calc(weight,doc[i],k)]
    score.sort(key = sortSecond, reverse = True)

    for i in range(len(score)):
        if (score[i][1]==0.0):
            del score[i:]
            break
    return score


# query_check calls query_docs and checks if we have atleast ten results or not, if not, then decreases value of k
def query_check_synonyms(query,new_dict,k,new_dict_name):
    tokens=nltk.word_tokenize(query)
    x=query_docs_synonyms(new_dict,query,len(new_dict_name),k)
    g=[]
    while (len(x)<10 and k>0.01):
        if(len(tokens)>1):
            k=k/2
            x=query_docs_synonyms(new_dict,query,len(new_dict_name),k)
        else:
            break
    x.sort(key = sortSecond, reverse = True)
    for i in x:
        g.append([new_dict_name[i[0]],i[1]])
    return g


# Function to check and correct all spellings in the query
def speller(query,new_dict):
    a=""
    for word in query.split(' '):
        if(spell.correction(word) in new_dict):
            a=a+" "+spell.correction(word)
        else:
            a=a+" "+word
    return a


# qury provides modularity to the code and we can see which features the functions include
def qury_with_spellcheck(query,new_dict,new_dict_name):
    query=speller(query,new_dict)
    return query_check(query,new_dict,0.5,new_dict_name)

def qury_with_equivalence_class(query,new_dict,new_dict_name):
    return query_check_synonyms(query,new_dict,0.5,new_dict_name)

def qury_with_equivalence_class_and_spellcheck(query,new_dict,new_dict_name):
    query=speller(query,new_dict)
    return query_check_synonyms(query,new_dict,0.5,new_dict_name)





# Function to make the zonal index for the title
def zonal_index(new_dict_name, zonal_ind):
    for i in new_dict_name:
        add_to_inverted_index(new_dict_name[i].lower(),zonal_ind,i)

# Function to score queries by giving weights to zones: body has 0.9 weight and title has 0.1 weight
def query_check_zone(query,new_dict,k,new_dict_name):
    tokens=nltk.word_tokenize(query)
    tok=[x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]
    token = [w.lower() for w in tok if w.isalpha()]
    x=query_docs(new_dict,query,len(new_dict),k)
    g=[]
# Heuristics to deal with cases when the number of results returned are less than ten
# If the number of cases are less than ten, we decrease the values of k by a factor of two.
    while (len(x)<10 and k>0.01):
        if(len(tokens)>1):
            k=k/2
            x=query_docs(new_dict,query,len(new_dict),k)
        else:
            break
    x.sort(key = sortSecond, reverse = True)
    zonal_ind={}
    zonal_index(new_dict_name,zonal_ind)
    score_title=score_zone_title(len(new_dict_name),query,len(tokens),zonal_ind,new_dict_name)
    for i in x:
        i[1]=0.7*i[1]+0.3*score_title[i[0]]
        g.append([new_dict_name[i[0]],i[1]])
    g.sort(key = sortSecond, reverse = True)
    return g


# Function to score queries by giving weights to zones: body has 0.8 weight and title has 0.2 weight and including the equivalence class feature
def query_check_zone_synonyms(query,new_dict,k,new_dict_name):
    tokens=nltk.word_tokenize(query)
    tok=[x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]
    token = [w.lower() for w in tok if w.isalpha()]
    x=query_docs_synonyms(new_dict,query,len(new_dict),k)
    g=[]
# Heuristics to deal with cases when the number of results returned are less than ten
# If the number of cases are less than ten, we decrease the values of k by a factor of two.
    while (len(x)<10 and k>0.01):
        if(len(tokens)>1):
            k=k/2
            x=query_docs_synonyms(new_dict,query,len(new_dict),k)
        else:
            break
    x.sort(key = sortSecond, reverse = True)
    zonal_ind={}
    zonal_index(new_dict_name,zonal_ind)
    score_title=score_zone_title(len(new_dict_name),query,len(tokens),zonal_ind,new_dict_name)
    for i in x:
        i[1]=0.7*i[1]+0.3*score_title[i[0]]
        g.append([new_dict_name[i[0]],i[1]])
    g.sort(key = sortSecond, reverse = True)
    return g
# calculate score of the title zone
def score_zone_title(No_of_docs,query,len_query,zonal_ind,new_dict_name):
    score=[0]*No_of_docs
    tokens=nltk.word_tokenize(query)
    for i in tokens:
        if i.lower() in zonal_ind:
            for j in zonal_ind[i.lower()]:
                score[j[0]]=score[j[0]]+j[1]
    for i in range(No_of_docs):
        score[i]=score[i]/(score[i]+len_query)
    return score

def qury_with_zone(query,new_dict,new_dict_name):
    return query_check_zone(query,new_dict,0.5,new_dict_name)
def qury_with_zone_and_spellcheck(query,new_dict,new_dict_name):
    query=speller(query,new_dict)
    return query_check_zone(query,new_dict,0.5,new_dict_name)
def qury_with_zone_and_spellcheck_and_equivalence_class(query,new_dict,new_dict_name):
    query=speller(query,new_dict)
    return query_check_zone_synonyms(query,new_dict,0.5,new_dict_name)




def dump_best_case(k5,query):
    b5 = k5[0:3]
    ret_docs = []
    for key in b5:
        # print(key)
        ret_docs.append(new_dict_document[key[0]])
    import json

    data = {"documents":ret_docs, "question": query}

    with open(top_3_docs, 'w') as f:
        json.dump(data, f)

# This function helps to see differences in the retrieved documents before and after the improvements of part two
def make_query(query):
    # print("Query is :",query,end="\n\n\n")
    # print("Corrected query by the spell checker :",speller(query,new_dict),end ="\n\n\n")
    index=['Title', 'Score']
    k1=qury(query,new_dict,new_dict_name)
    df1 = pd.DataFrame(k1, columns=index)
    # k2=qury_with_equivalence_class(query,new_dict,new_dict_name)
    # df2 = pd.DataFrame(k2, columns=index)
    # k3=qury_with_spellcheck(query,new_dict,new_dict_name)
    # df3 = pd.DataFrame(k3, columns=index)
    # k4=qury_with_zone(query,new_dict,new_dict_name)
    # df4 = pd.DataFrame(k4, columns=index)
    # k5=qury_with_zone_and_spellcheck_and_equivalence_class(query,new_dict,new_dict_name)
    # df5 = pd.DataFrame(k5, columns=index)
    # print("Output from part 1")
    # print(df1[0:10],end ="\n\n\n")
    # print("Output with equivalence classes")
    # print(df2[0:10],end ="\n\n\n")
    # print("Output with spellcheck")
    # print(df3[0:10],end ="\n\n\n")
    # print("Output with zone indexing")
    # print(df4[0:10],end ="\n\n\n")
    # print("Output with all three features: zone indexing, spellcheck and equivalence classes")
    # print(df5[0:10],end ="\n\n\n")

    temp_num = df1.Score[0]
    pos = k1
    # if(df2.size!=0 and temp_num<df2.Score[0]):
    #     pos = k2
    #     temp_num = df2.Score[0]
    # if(df3.size != 0 and temp_num<df3.Score[0]):
    #     pos = k3
    #     temp_num = df3.Score[0]
    # if(df4.size != 0 and temp_num<df4.Score[0]):
    #     pos = k4
    #     temp_num = df4.Score[0]
    # if(df5.size != 0 and temp_num<df5.Score[0]):
    #     pos = k5
    #     temp_num = df5.Score[0]
    dump_best_case(pos,query)
    return df1[0:1].Title

import json
with open(questions_file) as f:
  data_train = json.load(f)
counter = 0

from tqdm import tqdm
text = []
label = []
import nltk
from nltk import tokenize
for i in data_train['questions']:
    if(i['difficulty']=='hard_high_school' or i['difficulty'] == 'easy_high_school' or i['difficulty'] == 'regular_high_school' or i['difficulty'] =='national_high_school' or i['difficulty'] == 'HS'):
        label.append('HS')
        t1 = i['text']
        t1 = tokenize.sent_tokenize(t1)[-1]
        # print(t1)
        text.append(str(t1.encode('utf-8').decode('ascii',"ignore")))
    elif(i['difficulty']=='regular_college' or i['difficulty'] == 'hard_college' or i['difficulty'] == 'College'  or i['difficulty'] =='easy_college'):
        label.append('COL')
        t1 = i['text']
        t1 = tokenize.sent_tokenize(t1)[-1]
        # print(t1)
        text.append(str(t1.encode('utf-8').decode('ascii',"ignore")))


import csv
f = open(file_to_store_questions, "w+")
for i in range(len(text)):
    f.write("{}:::{}\n".format(text[i], label[i]))

for j in tqdm(range(start,start+total),desc = "Loop"):
    i = data_train['questions'][j]
    g = i['text'].encode('utf-8').decode('ascii',"ignore")
    ans_ret = make_query(g)
    ans_actual = i['page']
    for k in ans_ret:
        k = k.replace(" ", "_")
        if(ans_actual == k):
            counter+=1
            continue

print("accuracy:",counter/total)
