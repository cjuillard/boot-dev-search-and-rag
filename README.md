## Description
Repo for boot.dev course implementing a number of advanced search topics ([course link](https://www.boot.dev/courses/learn-retrieval-augmented-generation)). 

## Course outline
1. [X] Preprocecessing
  - tokenization
  - stop words
  - stemming
3. [X] TF-IDF
  - building an inverted index (token --> doc[])
  - building cache of Inverse Document Frequency (IDF)
  - build basic scoring function: TF-IDF
4. [ ] Keyword Search
  - implements [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) keyword search
    - The scoring function avoids some issues with term frequency saturation where a token appears many times in a document
    - Length of the document is also taken into consideration instead of just frequency of hits
5. [ ] Semantic Search
6. [ ] Chunking
7. [ ] Hybrid Search
8. [ ] LLMs
9. [ ] Reranking
10. [ ] Evaluation
11. [ ] Augmented Generation
12. [ ] Agentic
13. [ ] Multimodal

## Example usage
```bash
# Keyword search with Okapi BM25 scoring for "space adventure"
uv run cli/keyword_search_cli.py bm25search "space adventure"
```
## Tools used
- Python
