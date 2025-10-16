#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer
import pickle
import os

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
  
  def __add_document(self, doc_id, text):
    tokens = self.tokenizer.tokenize(text)
    for token in tokens:
      docsForToken = self.index.get(token, None)
      if docsForToken is None:
        docsForToken = []
        self.index[token] = docsForToken

      if doc_id not in docsForToken:
        docsForToken.append(doc_id)
      
      docsForToken.sort()
  
  def get_documents(self, term: str) -> list[int]:
    term = term.lower()
    if term not in self.index:
      return []
    return self.index[term]
  
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build inverted index")

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
            # print the search query here
            tokens = tokenizer.tokenize(args.query)
            print(f"Searching for: {args.query}")
            for movie in movies:
                movie_tokens = tokenizer.tokenize(movie["title"])
                if(check_partial_match(tokens, movie_tokens)):
                    matches.append(movie)
                    continue
            
            for i, match in enumerate(matches):
                print(f"{i+1}. {match['title']}")
        case "build":
            inverted_index = InvertedIndex(tokenizer)
            inverted_index.build(movies)
            inverted_index.save()

            docs = inverted_index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")

        case _:
            parser.print_help()


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