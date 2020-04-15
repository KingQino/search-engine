# -*- coding: utf-8 -*-
# @Time    : 2020/4/12 12:19 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : bm25_ctf_algorithm.py
# @Software: PyCharm
# @Data processing: Jingye Shang
# @Email: 592353879@qq.com
# Data source: TREC / Web Track /1997-2004 (document id : 301-700)
#              https://trec.nist.gov/data/webmain.html
# Reference:
#  1. https://github.com/Adonais0/Leetcode-Search-Engine
#  2. https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/


import nltk
import math
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Pre-processing document. Following the paper, stemming and stop-words removal are not
# recommended to use in the process.
def preprocessing_collection(collection):
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    stemming = PorterStemmer()
    preprocessed_collection = {}
    for document in list(collection):
        preprocessed_collection[document] = {}
        for field in list(collection[document]):
            # Convert all text to lower case
            collection[document][field] = collection[document][field].lower()

            # Tokenizing
            token_list = nltk.word_tokenize(collection[document][field])
            # taken only words (not punctuation and numbers)
            token_words = [w for w in token_list if w.isalpha()]

            # Stop-word Removal
            # stops = set(stopwords.words("english"))
            # meaningful_words = [w for w in token_words if not w in stops]

            # Stemming - it is not used in the paper, but I apply it.
            # stemmed_words = [stemming.stem(word) for word in meaningful_words]
            # stemmed_words = [stemming.stem(word) for word in token_words]

            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in token_words]

            preprocessed_collection[document][field] = lemmatized_words
    return preprocessed_collection


# The function is used to collect the statistical information of the document collection.
# It is the 1st version of the function, just consider the document as a whole,
# i.e. do not care about any field weights information in a document
def get_collection_statistics_v1(collection):
    corpus_terms = set()  # the terms set contains all the terms in the collection
    num_corpus_terms = 0  # the total number of terms in the collection
    num_docs = len(collection)  # the number of documents in the collection

    doc_terms = {}  # it is a dict type, and contains all the documents with terms
    doc_terms_unique = {}  # it is a dict type, contains all the documents, and each document has unique terms set
    doc_length = {}  # it is a dict type, and contains the length of each document

    for doc in collection:
        doc_terms_duplicate = []
        doc_len = 0

        # dont care about fields information, just consider a document as a whole
        for field in collection[doc]:
            num_corpus_terms = num_corpus_terms + len(collection[doc][field])
            corpus_terms = corpus_terms | set(collection[doc][field])
            doc_terms_duplicate = doc_terms_duplicate + collection[doc][field]
            doc_len = doc_len + len(collection[doc][field])

        doc_terms[doc] = doc_terms_duplicate
        doc_terms_unique[doc] = set(doc_terms[doc])
        doc_length[doc] = doc_len

    avg_doc_len = num_corpus_terms / num_docs  # the average document number

    doc_freq = {}  # the document frequency (df) for each term
    coll_term_freq = {}  # the collection term frequency (ctf) for each term
    term_freq = {}

    for word in corpus_terms:
        doc_freq[word] = 0
        coll_term_freq[word] = 0
        term_freq[word] = {}
        for doc_ in collection:
            term_freq[word][doc_] = 0
            if word in doc_terms[doc_]:
                doc_freq[word] = doc_freq[word] + 1
            for field in collection[doc_]:
                coll_term_freq[word] = coll_term_freq[word] + collection[doc_][field].count(word)
                term_freq[word][doc_] = term_freq[word][doc_] + collection[doc_][field].count(word)

    statistics = {}
    statistics['num_docs'] = num_docs
    statistics['corpus_terms'] = corpus_terms
    statistics['num_corpus_terms'] = num_corpus_terms
    statistics['doc_terms'] = doc_terms
    statistics['doc_terms_unique'] = doc_terms_unique
    statistics['doc_length'] = doc_length
    statistics['avg_doc_len'] = avg_doc_len
    statistics['doc_freq'] = doc_freq
    statistics['coll_term_freq'] = coll_term_freq
    statistics['term_freq'] = term_freq
    return statistics


# The 2nd version of the function, consider the document as a weighted field combination
def get_collection_statistics_v2(collection):
    return {}


# The function is used to collect the statistical information of the query.
def get_query_statistics(query):
    query_words = query.lower().split()

    query_length = len(query_words)
    query_term_freq = {}
    for qw in query_words:
        query_term_freq[qw] = query_words.count(qw)

    query_meta = {'query_length': query_length, 'query_term_freq': query_term_freq, 'query_words': query_words,
                  'query': query}
    return query_meta


# According to the query, do search in the document collection using the specified ranker (algorithm)
# query: query, meta: the statistics of the document collection, ranker: the model of a search algorithm
def do_search(query, meta, ranker):
    d_scores = {}  # store the relevant score for each query
    query_meta = get_query_statistics(query)
    # print("query:", query_meta['query'])
    for i in range(1, meta['num_docs'] + 1):
        d_scores[i] = [ranker.score(query_meta, str(i + 300), meta)]
    ranking_list = pd.DataFrame(d_scores).T
    ranking_list.columns = ['score']
    ranking_list.index = range(301, 301+meta['num_docs'])
    ranking_list = ranking_list.sort_values(by='score', ascending=False)

    return ranking_list, d_scores


# The implementation of BM25 algorithm, which is used to compare with BM25-CTF model
class bm25:

    def __init__(self, k1=1.25, b=0.9, k3=500):
        self.k1 = k1
        self.b = b
        self.k3 = k3

    def score_one(self, w, query_meta, meta, document_id):
        k1 = self.k1
        b = self.b
        k3 = self.k3

        K = k1 * ((1 - b) + b * (meta['doc_length'][document_id] / meta['avg_doc_len']))

        idf = math.log((meta['num_docs'] - meta['doc_freq'][w] + 0.5) / (meta['doc_freq'][w] + 0.5))
        tf_doc = ((k1 + 1) * meta['term_freq'][w][document_id]) / (K + meta['term_freq'][w][document_id])
        tf_query = ((k3 + 1) * query_meta['query_term_freq'][w]) / (k3 + query_meta['query_term_freq'][w])

        res = idf * tf_doc * tf_query

        return res

    def score(self, query_meta, document_id, meta):
        words = set(query_meta['query_words']) & meta['doc_terms_unique'][document_id]
        score = 0
        for w in words:
            score += self.score_one(w, query_meta, meta, document_id)
        return score


# The implementation of BM25-CTF algorithm
class bm25_ctf:

    def __init__(self, k1=1.25, b=0.9, k3=500, model_version=3):
        self.k1 = k1
        self.b = b
        self.k3 = k3
        self.model_version = model_version

    # compute the score of a term, which appears in query and document both (intersection)
    def score_one(self, w, query_meta, meta, document_id):
        k1 = self.k1
        b = self.b
        k3 = self.k3
        model_version = self.model_version

        K = k1 * ((1 - b) + b * (meta['doc_length'][document_id] / meta['avg_doc_len']))

        # compute the boosted idf component
        ictf = math.log(meta['num_corpus_terms'] / meta['coll_term_freq'][w])
        pidf = math.log(-(1 - math.exp(-meta['coll_term_freq'][w] / meta['num_docs'])) / math.log(
            1 - meta['doc_freq'][w] / meta['num_docs']) + 1)
        idf = math.log((meta['num_docs'] - meta['doc_freq'][w] + 0.5) / (meta['doc_freq'][w] + 0.5))
        bidf = ictf * pidf * idf

        # compute the boosted tf(d,q) component
        sum_ctf = sum([meta['coll_term_freq'][w] for w in meta['doc_terms_unique'][document_id]])
        porp_len = meta['doc_length'][document_id] - len(meta['doc_terms_unique'][document_id])
        tf_es = 1 + (meta['coll_term_freq'][w] / sum_ctf) * porp_len
        sum_ratio = sum(
            [meta['term_freq'][w][document_id] / (1 + (meta['coll_term_freq'][w] / sum_ctf) * porp_len) for w in
             meta['doc_terms_unique'][document_id]])
        C = meta['doc_length'][document_id] / sum_ratio
        btf = C * (meta['term_freq'][w][document_id] / tf_es)

        tf_query = ((k3 + 1) * query_meta['query_term_freq'][w]) / (k3 + query_meta['query_term_freq'][w])

        # when the model_version equals 1, the model just boost the idf;
        # version = 2, the model just boost the tf(d,q);
        # version = 3, the model will boost both of them above.
        if model_version == 1:
            tf_doc = ((k1 + 1) * meta['term_freq'][w][document_id]) / (K + meta['term_freq'][w][document_id])
            res = bidf * tf_doc * tf_query
        elif model_version == 2:
            tf_doc = ((k1 + 1) * btf) / (K + btf)
            res = idf * tf_doc * tf_query
        else:
            tf_doc = ((k1 + 1) * btf) / (K + btf)
            res = bidf * tf_doc * tf_query

        return res

    def score(self, query_meta, document_id, meta):
        words = set(query_meta['query_words']) & meta['doc_terms_unique'][document_id]
        score = 0
        for w in words:
            score += self.score_one(w, query_meta, meta, document_id)
        return score
