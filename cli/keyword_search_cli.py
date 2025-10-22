#!/usr/bin/env python3

import argparse
import json
import string
from nltk.stem import PorterStemmer
from search_utils import Tokenizer, InvertedIndex, sanitize_string, BM25_K1

def load_tokenizer() -> Tokenizer:
   stopwords_file = open("data/stopwords.txt")
   stopwords = stopwords_file.read().splitlines()
   stopwords = [sanitize_string(word) for word in stopwords]

   stemmer = PorterStemmer()
   tokenizer = Tokenizer(stopwords, stemmer)
   return tokenizer

def load_index_data() -> tuple[InvertedIndex, Tokenizer]:
   tokenizer = load_tokenizer()
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

    bm25_tf_parser = subparsers.add_parser(
      "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")

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
            tokenizer = load_tokenizer()
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
        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
        case _:
            parser.print_help()

def bm25_idf_command(term: string) -> float:
   tokenizer, inverted_index = load_index_data()
   return inverted_index.get_bm25_idf(term)

def bm25_tf_command(doc_id: int, term: string, k1: float = BM25_K1) -> float:
   tokenizer, inverted_index = load_index_data()
   return inverted_index.get_bm25_tf(doc_id, term, k1)

# Check if any search token is in any target token
def check_partial_match(search_tokens: list[str], target_tokens: list[str]) -> bool:
    for search_token in search_tokens:
      for target_token in target_tokens:
          if(search_token in target_token):
              return True
    return False

if __name__ == "__main__":
    main()