import string
from collections import Counter
from nltk.stem import PorterStemmer
import pickle
import os
import math


translation_table = str.maketrans("", "", string.punctuation)

def sanitize_string(s: str) -> str:
    s = s.lower()
    s = s.translate(translation_table)
    return s

def tokenize_string(s: str) -> list[str]:
    tokens = s.split()
    return tokens


class Tokenizer:
  def __init__(self, stopwords: list[str], stemmer: PorterStemmer):
    self.stopwords = stopwords
    self.stemmer = stemmer
  
  def tokenize(self, text: str) -> list[str]:
    tokens = tokenize_string(sanitize_string(text))
    tokens = [token for token in tokens if token not in self.stopwords]
    tokens = [self.stemmer.stem(token) for token in tokens]
    return tokens
  
class InvertedIndex:
  def __init__(self, tokenizer: Tokenizer):
    self.tokenizer = tokenizer
    self.index = {}
    self.docmap = {}
    self.term_frequencies = {}
  
  def __add_document(self, doc_id, text):
    tokens = self.tokenizer.tokenize(text)

    counter = Counter()
    self.term_frequencies[doc_id] = counter

    for token in tokens:
      counter[token] += 1
      docsForToken = self.index.get(token, None)
      if docsForToken is None:
        docsForToken = []
        self.index[token] = docsForToken

      if doc_id not in docsForToken:
        docsForToken.append(doc_id)
      
      docsForToken.sort()
  
  def get_documents(self, term: str) -> list[int]:
    token = self.tokenizer.tokenize(term)[0]
    return self.get_documents_for_token(token)
  
  def get_documents_for_token(self, token: str) -> list[int]:
    if token not in self.index:
      return []
    return self.index[token]
  
  def get_tf(self, doc_id: str, term: str) -> int:
    tokens = self.tokenizer.tokenize(term)
    if(len(tokens) > 1):
       raise ValueError("Term must be a single token")
    token = tokens[0]
    if doc_id not in self.term_frequencies:
      return 0
    if token not in self.term_frequencies[doc_id]:
      return 0
    return self.term_frequencies[doc_id][token]

  def get_idf(self, term: str) -> float:
    docs_for_term = self.get_documents(term)
    term_doc_count = len(docs_for_term)
    total_doc_count = len(self.docmap.keys())

    return math.log((total_doc_count + 1) / (term_doc_count + 1))

  def get_tfidf(self, doc_id: str, term: str) -> float:
    tf = self.get_tf(doc_id, term)
    idf = self.get_idf(term)
    return tf * idf
  
  def get_bm25_idf(self, term: str) -> float:
     tokens = self.tokenizer.tokenize(term)
     if(len(tokens) > 1):
       raise ValueError("Term must be a single token")
     token = tokens[0]

     docs_for_term = self.get_documents_for_token(token)
     term_doc_count = len(docs_for_term)
     total_doc_count = len(self.docmap.keys())

     return math.log((total_doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
  
  def build(self, movies):
    for i, movie in enumerate(movies):
       doc_id = i + 1
       context = f"{movie['title']} {movie['description']}"
       self.__add_document(doc_id, context)
       self.docmap[doc_id] = movie
  
  def save(self):
     os.makedirs("cache", exist_ok=True)
     pickle.dump(self.index, open("cache/index.pkl", "wb"))
     pickle.dump(self.docmap, open("cache/docmap.pkl", "wb"))
     pickle.dump(self.term_frequencies, open("cache/term_frequencies.pkl", "wb"))
  
  def load(self):
    self.index = pickle.load(open("cache/index.pkl", "rb"))
    self.docmap = pickle.load(open("cache/docmap.pkl", "rb"))
    self.term_frequencies = pickle.load(open("cache/term_frequencies.pkl", "rb"))

