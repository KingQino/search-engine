# -*- coding: utf-8 -*-
# @Time    : 2020/4/12 11:34 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : main.py
# @Software: PyCharm
# @Data processing: Jingye Shang
# @Email: 592353879@qq.com
# Data source: TREC / Web Track /1997-2004 (document id : 301-700)
#              https://trec.nist.gov/data/webmain.html
# Reference:
#  1. https://github.com/Adonais0/Leetcode-Search-Engine
#  2. https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/


import json
import pandas as pd
from bm25_ctf_algorithm import preprocessing_collection
from bm25_ctf_algorithm import get_collection_statistics_v1
from bm25_ctf_algorithm import bm25_ctf, bm25
from bm25_ctf_algorithm import do_search


# read files that we will use
try:
    cache_file = open('collection.json', 'r')
    cache_contents = cache_file.read()
    collection = json.loads(cache_contents)
    cache_file.close()
except:
    print("something bad happens!")

# pre-process the document collection and get the statistics of the collection
preprocessed_coll = preprocessing_collection(collection)
coll_sta = get_collection_statistics_v1(preprocessed_coll)


query_list = ["Crime",
              "illegal activity",
              "Poliomyelitis disease",
              "Hubble telescope",
              "Ireland consular information sheet",
              "Citizen attitudes toward prairie dogs",
              "JPL stardust comet wild",
              "American music",
              "oil petroleum resources",
              "child care"]
# select the ranking model
# Note that there are 3 kinds of combination for bm25-ctf model,
# 1st, just boost the idf using ctf
# 2st, just boost the tf between query and document using ctf
# 3rd, boost both of the components above.
# for example, ranker = bm25_ctf(k1 =1.1, b =0.8, k3 =400, model_version =1)
# In addition, you can tune parameters (k1, b, k3)of bm25
ranker = bm25_ctf(model_version=3)
# ranker = bm25()
for query in query_list:
    print("query:",query)
    ranking_list, _ = do_search(query, coll_sta, ranker)
    print(ranking_list.head(10))
    print("_________________________________________")

#################################################################
######################## Evaluation #############################
#################################################################

labels = pd.read_csv('Labels.csv',dtype='Int64')
ranker = bm25_ctf(model_version=3)
# ranker = bm25()
sum_AP = 0
for i,q in enumerate(query_list, start=1):
   print("query:", q)
   ranking_list, _ = do_search(q, coll_sta, ranker)
   label_true = [e+301 for e in list(labels[labels[str(i)]==1].index)]
   id_pred = list(ranking_list.head(20).index.values)
   print("label size:", len(label_true))
   P_10 = 0
   num_rev = 0
   sum_P_j = 0
   for j, id_ in enumerate(id_pred, start=1):
       if id_ in label_true:
           P_10 += 1
           num_rev += 1
       if j==10:
           print("P@10:", P_10 / 10)
       P_j = num_rev/j
       sum_P_j += P_j
   AP = sum_P_j/len(id_pred)
   sum_AP += AP
# print('AP@20:',AP)
MAP = sum_AP/len(query_list)
print('\nMAP@20', MAP)
