# Raj Sanjay Shah
#
# Please change the file paths as required
address_of_dictionary="dict.txt"
address_of_docnumber_title_mapping="dict_name.txt"
address_of_vocabulary="vocab.txt"
document_text="document.txt"
file_name_documents='wiki_lookup.json'

import re,nltk,collections
import sys
from nltk.util import ngrams
from bs4 import BeautifulSoup as BSoup
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
import string
import pickle
import json

with open(file_name_documents) as f:
  data = json.load(f)
lis_doc =[]
for key in data:
    lis_doc.append(data[key]['text'])

def add_to_inverted_index(text,dic,docnumber,dic_name,vocab,doc_text):
# Removing punctuation and converting everything to lower case
    tokens=nltk.word_tokenize(text)
    tok=[x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]
    token = [w.lower() for w in tok if w.isalpha()]

# Mapping doc_number to title
    a=text.split('\n')[0]
    dic_name[docnumber]=a
    # print(a)
    doc_text[a] = text
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
            vocab.append(a)
            dic[a]=[[docnumber,1]]

# myDict contains the inverted index
# myDictName maps all document numbers to titles
doc_text = {}
myDict={}
myDictName={}
vocab=[]
for i in range(len(lis_doc)):
    add_to_inverted_index(lis_doc[i],myDict,i,myDictName,vocab,doc_text)

f = open(address_of_dictionary,"wb")
with f as filehandle:
    pickle.dump(myDict, filehandle)
f.close()

f = open(document_text,"wb")
with f as filehandle:
    pickle.dump(doc_text, filehandle)
f.close()



f = open(address_of_docnumber_title_mapping,"wb")
with f as filehandle:
    pickle.dump(myDictName, filehandle)
f.close()

f = open(address_of_vocabulary,"wb")
with f as filehandle:
    pickle.dump(vocab, filehandle)
f.close()
