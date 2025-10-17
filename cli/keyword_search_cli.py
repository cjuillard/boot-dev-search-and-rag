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

    args = parser.parse_args()

    movies = json.load(open("data/movies.json"))["movies"]

    
    stopwords_file = open("data/stopwords.txt")
    stopwords = stopwords_file.read().splitlines()
    stopwords = [sanitize_string(word) for word in stopwords]

    stemmer = PorterStemmer()
    tokenizer = Tokenizer(stopwords, stemmer)

    matches = []
    match args.command:
        case "search":
            tokens = tokenizer.tokenize(args.query)
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.load()

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
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.build(movies)
            inverted_index.save()

            doc_ids = inverted_index.get_documents("merida")
            print(f"First document for token 'merida' = {doc_ids[0]}")
        case "tf":
            inverted_index = load_inverted_index(tokenizer)
            tf = inverted_index.get_tf(args.doc_id, args.term)
            print(f"{tf}")
        case "idf":
            inverted_index = load_inverted_index(tokenizer)
            docs_for_term = inverted_index.get_documents(args.term)
            term_doc_count = len(docs_for_term)
            total_doc_count = len(inverted_index.docmap.keys())
            
            idf = math.log((total_doc_count + 1) / (term_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()

def load_inverted_index(tokenizer: Tokenizer) -> InvertedIndex:
    inverted_index = InvertedIndex(tokenizer)
    inverted_index.load()
    return inverted_index

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