import string
from collections import Counter
from nltk.stem import PorterStemmer
import pickle
import os
import math

# Parameter to increase or decrease the importance of a term's frequency as the frequency increases.
# Larger values mean less falloff in importance as the frequency increases.
BM25_K1 = 1.5
# Parameter to control the importance of the document length (normalization strength). Range of [0, 1]
# Larger values mean less importance is given to longer documents.
BM25_B = 0.75

CACHE_DIR = "cache"

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
    self.document_lengths = {}
  
  def __add_document(self, doc_id, text):
    tokens = self.tokenizer.tokenize(text)

    counter = Counter()
    self.term_frequencies[doc_id] = counter
    self.document_lengths[doc_id] = len(tokens)

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
  
  def __get_avg_doc_length(self) -> float:
    if len(self.document_lengths) == 0:
      return 0
    
    total_length = 0
    for doc_id in self.document_lengths:
      total_length += self.document_lengths[doc_id]
    return total_length / len(self.document_lengths)

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
     if len(tokens) != 1:
       raise ValueError("Term must be a single token")
     token = tokens[0]

     docs_for_term = self.get_documents_for_token(token)
     term_doc_count = len(docs_for_term)
     total_doc_count = len(self.docmap)

     return math.log((total_doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
  
  def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    tf = self.get_tf(doc_id, term)

    # Length normalization factor
    avg_doc_length = self.__get_avg_doc_length()
    doc_length = self.document_lengths.get(doc_id, 0)
    if avg_doc_length == 0:
      length_norm = 1
    else:
      length_norm = 1 - b + b * (doc_length / avg_doc_length)

    return (tf * (k1 + 1)) / (tf + k1 * length_norm)
  
  def build(self, movies):
    for i, movie in enumerate(movies):
       doc_id = i + 1
       context = f"{movie['title']} {movie['description']}"
       self.__add_document(doc_id, context)
       self.docmap[doc_id] = movie
  
  def save(self):
     os.makedirs(CACHE_DIR, exist_ok=True)
     pickle.dump(self.index, open(f"{CACHE_DIR}/index.pkl", "wb"))
     pickle.dump(self.docmap, open(f"{CACHE_DIR}/docmap.pkl", "wb"))
     pickle.dump(self.term_frequencies, open(f"{CACHE_DIR}/term_frequencies.pkl", "wb"))
     pickle.dump(self.document_lengths, open(f"{CACHE_DIR}/document_lengths.pkl", "wb"))
  
  def load(self):
    self.index = pickle.load(open(f"{CACHE_DIR}/index.pkl", "rb"))
    self.docmap = pickle.load(open(f"{CACHE_DIR}/docmap.pkl", "rb"))
    self.term_frequencies = pickle.load(open(f"{CACHE_DIR}/term_frequencies.pkl", "rb"))
    self.document_lengths = pickle.load(open(f"{CACHE_DIR}/document_lengths.pkl", "rb"))
  
  def bm25(self, doc_id, term) -> float:
    idf = self.get_bm25_idf(term)
    tf = self.get_bm25_tf(doc_id, term)
    return tf * idf
  
  def bm25_search(self, query, limit) -> list[any]:
    tokens = self.tokenizer.tokenize(query)
    scores = {}
    for doc_id in self.docmap:
      score = 0
      for token in tokens:
        score += self.bm25(doc_id, token)
      scores[doc_id] = score

    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]
    return [{'doc': self.docmap[doc_id], 'score': score} for doc_id, score in sorted_scores]
