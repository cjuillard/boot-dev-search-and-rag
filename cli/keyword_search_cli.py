#!/usr/bin/env python3

import argparse
import json
import string

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    movies = json.load(open("data/movies.json"))["movies"]

    matches = []
    match args.command:
        case "search":
            # print the search query here
            tokens = tokenize_string(sanitize_string(args.query))
            print(f"Searching for: {args.query}")
            for movie in movies:
                movie_tokens = tokenize_string(sanitize_string(movie["title"]))
                if(check_partial_match(tokens, movie_tokens)):
                    matches.append(movie)
                    continue
            
            for i, match in enumerate(matches):
                print(f"{i+1}. {match['title']}")
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