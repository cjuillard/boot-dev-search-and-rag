#!/usr/bin/env python3

import argparse
import json
import string
from collections import Counter
from nltk.stem import PorterStemmer
import pickle
import os
import math
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

def load_index_data() -> tuple[InvertedIndex, Tokenizer]:
   stopwords_file = open("data/stopwords.txt")
   stopwords = stopwords_file.read().splitlines()
   stopwords = [sanitize_string(word) for word in stopwords]

   stemmer = PorterStemmer()
   tokenizer = Tokenizer(stopwords, stemmer)
   inverted_index = InvertedIndex(tokenizer)
   inverted_index.load()
   return tokenizer, inverted_index

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "search":
            tokenizer, inverted_index = load_index_data()
            tokens = tokenizer.tokenize(args.query)
            print(f"Searching for: {args.query}")

            doc_ids = []
            for token in tokens:
                currDocs = inverted_index.get_documents(token)
                for doc in currDocs:
                    if doc not in doc_ids:
                        doc_ids.append(doc)
                        if len(doc_ids) >= 5:
                           break
            

            for doc in doc_ids:
              movie = inverted_index.docmap[doc]
              print(f"{doc} - {movie['title']}")
              
        case "build":
            movies = json.load(open("data/movies.json"))["movies"]
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.build(movies)
            inverted_index.save()

            doc_ids = inverted_index.get_documents("merida")
            print(f"First document for token 'merida' = {doc_ids[0]}")
        case "tf":
            tokenizer, inverted_index = load_index_data()
            tf = inverted_index.get_tf(args.doc_id, args.term)
            print(f"{tf}")
        case "idf":
            tokenizer, inverted_index = load_index_data()
            idf = inverted_index.get_idf(args.term)
            
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tokenizer, inverted_index = load_index_data()
            tf_idf = inverted_index.get_tfidf(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case _:
            parser.print_help()

def bm25_idf_command(term: string) -> float:
   tokenizer, inverted_index = load_index_data()
   return inverted_index.get_bm25_idf(term)

translation_table = str.maketrans("", "", string.punctuation)
def sanitize_string(s: str) -> str:
    s = s.lower()
    s = s.translate(translation_table)
    return s

def tokenize_string(s: str) -> list[str]:
    tokens = s.split()
    return tokens
    # tokens

# Check if any search token is in any target token
def check_partial_match(search_tokens: list[str], target_tokens: list[str]) -> bool:
    for search_token in search_tokens:
      for target_token in target_tokens:
          if(search_token in target_token):
              return True
    return False

if __name__ == "__main__":
    main()