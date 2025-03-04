import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import time
import math
import unicodedata
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize
BASE_URL = "https://en.wikipedia.org"
START_PAGE = "/wiki/Python_(programming_language)"
visited = set()
queue = [START_PAGE]
inverted_index = defaultdict(set)  # word -> set of pages
document_word_count = defaultdict(lambda: defaultdict(int))  # page -> word -> count
doc_lengths = defaultdict(int)  # page -> total words
total_docs = 0

# Pre-processing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in tokens]

# Get Wikipedia links
def is_wikipedia_url(url):
    parsed_url = urlparse(url)
    return parsed_url.netloc == "en.wikipedia.org" and parsed_url.path.startswith("/wiki/")

def get_wikipedia_links(page_url):
    full_url = urljoin(BASE_URL, page_url)
    response = requests.get(full_url)
    if response.status_code != 200:
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    links = set()
    for link in soup.find_all("a", href=True):
        href = link.get("href")
        if href and is_wikipedia_url(urljoin(BASE_URL, href)):
            links.add(urlparse(href).path)
    return links

# Index page content
def index_page(page_url):
    global total_docs
    full_url = urljoin(BASE_URL, page_url)
    response = requests.get(full_url)
    if response.status_code != 200:
        return

    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text()
    tokens = clean_text(text)

    doc_word_count = defaultdict(int)
    for word in tokens:
        doc_word_count[word] += 1
        inverted_index[word].add(page_url)

    document_word_count[page_url] = doc_word_count
    doc_lengths[page_url] = len(tokens)
    total_docs += 1

# TF-IDF score calculation
def tf_idf(term, doc):
    tf = document_word_count[doc][term] / doc_lengths[doc]
    df = len(inverted_index[term])
    idf = math.log(total_docs / (1 + df))
    return tf * idf

# Crawl Wikipedia pages and build the index
def crawl_wikipedia(max_pages=100):
    pages_crawled = 0
    while queue and pages_crawled < max_pages:
        current_page = queue.pop(0)
        if current_page in visited:
            continue

        print(f"Crawling: {BASE_URL}{current_page}")
        visited.add(current_page)
        index_page(current_page)
        pages_crawled += 1

        for link in get_wikipedia_links(current_page):
            if link not in visited:
                queue.append(link)

        time.sleep(1)

    print(f"\nCrawling complete! Total pages visited: {pages_crawled}")

# Search using TF-IDF ranking
def search(query, top_n=5):
    query_tokens = clean_text(query)
    scores = defaultdict(float)

    for term in query_tokens:
        if term in inverted_index:
            for doc in inverted_index[term]:
                scores[doc] += tf_idf(term, doc)

    ranked_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop {top_n} results for '{query}':\n")
    for doc, score in ranked_results[:top_n]:
        print(f"{BASE_URL}{doc} (score: {score:.4f})")

# Run the crawler and search engine
crawl_wikipedia(max_pages=50)
search("programming language")
