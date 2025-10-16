#!/usr/bin/env python3

import argparse
import json


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
            print(f"Searching for: {args.query}")
            for movie in movies:
                if args.query in movie["title"]:
                    matches.append(movie)
            
            for i, match in enumerate(matches):
                print(f"{i+1}. {match["title"]}")

            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()